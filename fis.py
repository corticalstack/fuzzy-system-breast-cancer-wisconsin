import time
from contextlib import contextmanager
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import skfuzzy as fz
from skfuzzy import control as ct
from sklearn.metrics import accuracy_score
from scipy import stats
from matplotlib import pyplot as plt

# Skfuzzy API
# https://scikit-fuzzy.readthedocs.io/en/latest/api/skfuzzy.html

# Example Skfuzzy FIS
#https://loctv.wordpress.com/2016/10/02/using-scikit-fuzzy-to-build-an-obesity-diagnosis-system/

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
        self.X_y_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_predict = []
        self.number_tests = 2
        self.ant = {}
        self.diagnosis = None
        self.rules = []
        self.accuracy_score = 0
        self.system = None

        #self.tests = [{'PerimeterMax': 'low', 'mf': 'trimf'}, {'trimf': 'high'}]

        with timer('\nLoad dataset'):
            self.load_data()

        with timer('\nSplit dataset'):
            self.set_y()
            self.remove_target_from_X()
            self.train_test_split()
            self.set_X_y_train()

        with timer('\nSetting Antecedents Universe'):
            pass

        for i in range(self.number_tests):
            self.ant = {}
            self.diagnosis = None
            self.rules = []
            self.accuracy_score = 0
            self.create_antecendents_universe()
            with timer('\nSetting Antecedent MFs'):
                self.set_test_antecendents_mfs(i)

            with timer('\nSetting Consequent'):
                self.set_test_consequent_mfs(i)

            with timer('\nSetting Rules'):
                self.set_test_rules(i)

            with timer('\nSetting System'):
                self.system = ct.ControlSystem(rules=self.rules)
                self.sim = ct.ControlSystemSimulation(self.system)

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

    def set_X_y_train(self):
        # Used for MF shaping when feature distribution filtered by target
        self.X_y_train = pd.concat([self.X_train, self.y_train], axis=1)

    def print_shape(self):
        print('\tRow count:\t', '{}'.format(self.X.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.X.shape[1]))

    def create_antecendents_universe(self):
        # Set universe boundary for each feature
        for feat in self.X:
            if feat == 'ID':
                continue
            self.ant[feat] = ct.Antecedent(np.linspace(self.X[feat].min(), self.X[feat].max(), num=200), feat)

    def set_test_antecendents_mfs(self, test):
        # Diagnosis: Benign = class 0, Malignant = Class 1
        for a in self.ant:
            ant_mfs_func = 'set_antecedents_mfs_test_' + str(test)
            getattr(self, ant_mfs_func)(a)

    def set_antecedents_mfs_test_0(self, a):
        mean_0 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 0, a].mean()
        std_0 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 0, a].std()
        mean_1 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 1, a].mean()
        std_1 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 1, a].std()

        # Low risk
        self.ant[a]['low'] = fz.gaussmf(self.ant[a].universe, mean_0, std_0)

        # High risk
        self.ant[a]['high'] = fz.gaussmf(self.ant[a].universe, mean_1, std_1)

    def set_antecedents_mfs_test_1(self, a):
        mean_0 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 0, a].mean()
        std_0 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 0, a].std()
        mean_1 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 1, a].mean()
        min_1 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 1, a].min()
        max_1 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 1, a].max()
        std_1 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 1, a].std()
        peak_1 = self.feature_kde_peak(a, self.X_y_train, 1)
        q25_1 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 1, a].quantile(.25)
        q75_1 = self.X_y_train.loc[self.X_y_train['Diagnosis'] == 1, a].quantile(.75)

        # Low risk
        self.ant[a]['low'] = fz.gaussmf(self.ant[a].universe, mean_0, std_0)

        # High risk
        from scipy import stats
        kernel = stats.gaussian_kde(self.X_y_train.loc[self.X_y_train['Diagnosis'] == 1, a])
        x = np.linspace(self.X[a].min(), self.X[a].max(), num=200)
        kernel = kernel(x)
        peak = x[np.argsort(kernel)[-1]]

       # self.ant[a]['high'] = fz.gaussmf(self.ant[a].universe, mean_1, std_1)
        #self.ant[a]['high'] = fz.trimf(self.ant[a].universe, [min_1, peak_1, max_1])
        self.ant[a]['high'] = fz.trapmf(self.ant[a].universe, [min_1, q25_1, q75_1, max_1])
        self.ant[a].view()

    def feature_mean(self, feat, df, target):
        return df.loc[df['Diagnosis'] == target, feat].mean()

    def feature_max(self, feat, df, target):
        return df.loc[df['Diagnosis'] == target, feat].max()

    def feature_quantile(self, feat, df, target, q):
        return df.loc[df['Diagnosis'] == target, feat].quantile(q)

    def feature_kde_peak(self, feat, df, target):
        kernel = stats.gaussian_kde(df.loc[df['Diagnosis'] == target, feat])
        universe = np.linspace(df[feat].min(), df[feat].max(), num=200)
        kernel = kernel(universe)
        return universe[np.argsort(kernel)[-1]]

    def set_test_consequent_mfs(self, test):
        con_mfs_func = 'set_consequent_mfs_test_' + str(test)
        getattr(self, con_mfs_func)()

    def set_consequent_mfs_test_0(self):
        self.diagnosis = ct.Consequent(np.arange(0, 100, 1), 'diagnosis')
        self.diagnosis['benign'] = fz.trapmf(self.diagnosis.universe, [0, 10, 40, 50])
        self.diagnosis['malignant'] = fz.trapmf(self.diagnosis.universe, [50, 60, 90, 100])

    def set_consequent_mfs_test_1(self):
        self.diagnosis = ct.Consequent(np.arange(0, 100, 1), 'diagnosis')
        self.diagnosis['benign'] = fz.trapmf(self.diagnosis.universe, [0, 10, 40, 50])
        self.diagnosis['malignant'] = fz.trapmf(self.diagnosis.universe, [50, 60, 90, 100])

    def set_test_rules(self, test):
        rules_func = 'set_rules_test_' + str(test)
        getattr(self, rules_func)()

    def set_rules_test_0(self):
        r = ct.Rule(self.ant['PerimeterMax']['low'], consequent=self.diagnosis['benign'], label='Benign')
        self.rules.append(r)

        r = ct.Rule(self.ant['PerimeterMax']['high'], consequent=self.diagnosis['malignant'], label='Malignant')
        self.rules.append(r)

    def set_rules_test_1(self):
        r = ct.Rule(self.ant['PerimeterMax']['low'] & self.ant['ConcavePointsMax']['low'],
                      consequent=self.diagnosis['benign'], label='Benign')
        self.rules.append(r)

        r = ct.Rule(self.ant['PerimeterMax']['high'] & self.ant['ConcavePointsMax']['high'],
                      consequent=self.diagnosis['malignant'], label='Malignant')
        self.rules.append(r)

    def predict(self):
        id = 0
        y_pred = []
        for di, dr in self.X_test.iterrows():
            for si, sv in dr.iteritems():
                if si == 'ID':
                    id = sv
                for rule in self.rules:
                    rule_str = str(rule)
                    if si in rule_str:
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
        # JP see self.system.view_n() and see if useful, what it does

wdbcFis = WDBCFis()
