import time
from contextlib import contextmanager
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import skfuzzy as fz
from skfuzzy import control as ctrl
from matplotlib import pyplot as plt



@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class WDBCFis:
    def __init__(self):
        self.random_state = 20
        self.column_stats = {}
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.antecendants = {}
        self.diagnosis = None
        self.PerimeterMax = None
        self.ConcavePointsMax = None

        self.rule1 = None

        with timer('\nLoad dataset'):
            self.load_data()

        with timer('\nSplit dataset'):
            self.set_y()
            self.remove_target_from_X()
            self.train_test_split()

        with timer('\nSetting Antecedents'):
            self.set_antecendents()

        with timer('\nSetting Consequent'):
            self.set_consequent()

        with timer('\nSetting Membership Functions'):
            self.set_mf()

        with timer('\nSetting Rules'):
            self.set_rules()

        with timer('\nSetting System'):
            system = ctrl.ControlSystem(rules=[self.rule1])
            self.sim = ctrl.ControlSystemSimulation(system)

        with timer('\nMaking Predictions'):
            # Should lop through all test data here, store, defuzzify, compoare with ground truth
            self.sim.input['PerimeterMax'] = 150
            self.sim.compute()
            print('Decision is ', self.sim.output['diagnosis'])

    def load_data(self):
        self.X = pd.read_csv('data/wdbc_selected_cols.csv')

        print('\n', '_' * 40, 'Shape After Data Load', '_' * 40)
        self.print_shape()

    def set_y(self):
        self.y = self.X['Diagnosis']

    def remove_target_from_X(self):
        self.X.drop('Diagnosis', axis=1, inplace=True)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30,
                                                                                random_state=self.random_state)

    def print_shape(self):
        print('\tRow count:\t', '{}'.format(self.X.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.X.shape[1]))

    def set_antecendents(self):
        for col in self.X:
            if col == 'ID':
                continue
            # Set universe for each antecendant
            self.antecendants[col] = ctrl.Antecedent(np.linspace(self.X[col].min(), self.X[col].max()), col)
        #self.ConcavePointsMax = ctrl.Antecedent(np.arange(0, 100000, 1000), 'ConcavePointsMax')

    def set_consequent(self):
        self.diagnosis = ctrl.Consequent(np.arange(0, 100, 1), 'diagnosis')
        self.diagnosis['high'] = fz.trapmf(self.diagnosis.universe, [40, 55, 80, 100])

    def set_mf(self):
        for ant in self.antecendants:
            if ant == 'PerimeterMax':
                self.antecendants[ant]['low'] = fz.trapmf(self.antecendants[ant].universe, [0, 100, 150, 250])

    def set_rules(self):
        self.rule1 = ctrl.Rule(self.antecendants['PerimeterMax']['low'], consequent=self.diagnosis['high'], label='High Risk')


wdbcFis = WDBCFis()
