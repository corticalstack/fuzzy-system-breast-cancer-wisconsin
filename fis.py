import time
from contextlib import contextmanager
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import skfuzzy as fz
from skfuzzy import control as ctrl
from sklearn.metrics import accuracy_score
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
        self.y_predict = []

        self.ant = {}  # antecendents
        self.diagnosis = None
        self.PerimeterMax = None
        self.ConcavePointsMax = None

        self.rule1 = None
        self.rules = []

        self.accuracy_score = 0

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
            system = ctrl.ControlSystem(rules=self.rules)
            self.sim = ctrl.ControlSystemSimulation(system)

        with timer('\nMaking Predictions'):
            # Should lop through all test data here, store, defuzzify, compoare with ground truth
            self.predict()

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
            # Set universe for each antecendant - believe default intervals for linspace is 50
            self.ant[col] = ctrl.Antecedent(np.linspace(self.X[col].min(), self.X[col].max()), col)

    def set_consequent(self):
        self.diagnosis = ctrl.Consequent(np.arange(0, 100, 1), 'diagnosis')
        self.diagnosis['low'] = fz.trapmf(self.diagnosis.universe, [0, 10, 40, 50])
        self.diagnosis['high'] = fz.trapmf(self.diagnosis.universe, [50, 60, 90, 100])

    # Diagnosis: Malignant = Class 1, Benign = class 0
    def set_mf(self):
        df_tmp = pd.concat([self.X_train, self.y_train], axis=1)
        for a in self.ant:
                # Low risk
                mean = df_tmp.loc[df_tmp['Diagnosis'] == 0, a].mean()
                std = df_tmp.loc[df_tmp['Diagnosis'] == 0, a].std()
                self.ant[a]['low'] = fz.gaussmf(self.ant[a].universe, mean, std)

                # High risk
                mean = df_tmp.loc[df_tmp['Diagnosis'] == 1, a].mean()
                std = df_tmp.loc[df_tmp['Diagnosis'] == 1, a].std()
                self.ant[a]['high'] = fz.gaussmf(self.ant[a].universe, mean, std)
                #self.ant[a].view()

    def set_rules(self):
        r = ctrl.Rule(self.ant['PerimeterMax']['low'] & self.ant['ConcavePointsMax']['low'],
                      consequent=self.diagnosis['low'], label='Low Risk')
        self.rules.append(r)

        r = ctrl.Rule(self.ant['PerimeterMax']['high'] & self.ant['ConcavePointsMax']['high'],
                      consequent=self.diagnosis['high'], label='high Risk')
        self.rules.append(r)

    def predict(self):
        id = 0
        y_pred = []
        for di, dr in self.X_test.iterrows():
            for si, sv in dr.iteritems():
                if si == 'ID':
                    id = sv
                if si == 'PerimeterMax' or si == 'ConcavePointsMax':
                    self.sim.input[si] = sv
            self.sim.compute()
            output = {'ID': sv, 'FuzzyOut': self.sim.output['diagnosis'], 'CrispOut': 0 if self.sim.output['diagnosis'] < 50 else 1}
            y_pred.append(output['CrispOut'])
            self.y_predict.append(output)

        for r in self.y_predict:
            print(r)
        for y in self.y_test:
            print(y)
        self.accuracy_score = accuracy_score(self.y_test, y_pred)
        print('Accuracy ', self.accuracy_score)


wdbcFis = WDBCFis()
