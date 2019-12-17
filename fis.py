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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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
        self.astats = {}
        self.diagnosis = None
        self.rules = []
        self.accuracy_score = 0
        self.system = None
        self.max_iters = 100

        self.tests = [[{"Features": [{"PerimeterMax": {"mf": {"low": 'gaussmf',
                                                              "high": 'trimf'}}}]}],
                      [{"Features": [{"PerimeterMax": {"mf": {"low": 'gaussmf',
                                                              "high": 'gaussmf'}}},
                                     {"ConcavePointsMax": {"mf": {"low": 'gaussmf',
                                                                  "high": 'gaussmf'}}}]}]]
        self.test_cols = []

        with timer('\nLoad dataset'):
            self.load_data()

        with timer('\nSplit dataset'):
            self.set_y()
            self.remove_target_from_X()

        with timer('\nSetting Antecedents Universe'):
            for i, t in enumerate(self.tests):
                with timer('\nTest ' + str(i)):
                    self.set_test_cfg(t)

                with timer('\nPreparing dataset for test '):
                    self.train_test_split()
                    self.set_X_train_test_cols()
                    self.set_X_y_train()

                    self.ant = {}
                    self.diagnosis = None
                    self.rules = []
                    self.accuracy_score = 0

                with timer('\nCreating Antecedent Universe'):
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

    def set_test_cfg(self, t):
        for cfg in t:
            if 'Features' in cfg:
                for f in cfg['Features']:
                    for fk, fv in f.items():
                        if 'mf' in f[fk]:
                            for mfclass, mfshape in f[fk]['mf'].items():
                                self.test_cols.append({fk: (mfclass, mfshape)})

    def set_X_train_test_cols(self):
        cols = []
        for c in self.test_cols:
            for k, v in c.items():
                cols.append(k)
        cols = list(set(cols))
        self.X_train = self.X_train[cols]
        self.X_test = self.X_test[cols]

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
        for feat in self.X_train:
            self.ant[feat] = ct.Antecedent(np.linspace(self.X_train[feat].min(), self.X_train[feat].max(), num=200), feat)

    def set_test_antecendents_mfs(self, test):
        # Diagnosis: Benign = class 0, Malignant = Class 1
        for a in self.ant:
            s = self.set_antecedents_stats(a)
            self.set_antecedents_mfs_test(a, s)

    def set_antecedents_stats(self, a):
        s = {}
        s['min0'] = self.feature_min(a, self.X_y_train, 0)
        s['min1'] = self.feature_min(a, self.X_y_train, 1)

        s['max0'] = self.feature_max(a, self.X_y_train, 0)
        s['max1'] = self.feature_max(a, self.X_y_train, 1)

        s['mean0'] = self.feature_mean(a, self.X_y_train, 0)
        s['mean1'] = self.feature_mean(a, self.X_y_train, 1)

        s['std0'] = self.feature_std(a, self.X_y_train, 0)
        s['std1'] = self.feature_std(a, self.X_y_train, 1)

        s['q250'] = self.feature_quantile(a, self.X_y_train, 0, 0.25)
        s['q251'] = self.feature_quantile(a, self.X_y_train, 1, 0.25)

        s['q750'] = self.feature_quantile(a, self.X_y_train, 0, 0.75)
        s['q751'] = self.feature_quantile(a, self.X_y_train, 1, 0.75)

        s['pke0'] = self.feature_kde_peak(a, self.X_y_train, 0)
        s['pke1'] = self.feature_kde_peak(a, self.X_y_train, 1)

        return s

    def set_antecedents_mfs_test(self, a, s):
        for c in self.test_cols:
            for k, v in c.items():
                if k == a:
                    self.ant[a][v[0]] = getattr(self, 'mf_' + v[1])(a, v[0], s)
        self.ant[a].view()

    # def set_antecedents_mfs_test_0(self, a, s):
    #     #     # Low
    #     #     self.ant[a]['low'] = fz.gaussmf(self.ant[a].universe, s['mean0'], s['std0'])
    #     #
    #     #     # High
    #     #     self.ant[a]['high'] = fz.gaussmf(self.ant[a].universe, s['mean1'], s['std1'])
    #     #
    #     # def set_antecedents_mfs_test_1(self, a, s):
    #     #     # Low risk
    #     #     self.ant[a]['low'] = fz.gaussmf(self.ant[a].universe, s['mean0'], s['std0'])
    #     #
    #     #    # self.ant[a]['high'] = fz.gaussmf(self.ant[a].universe, mean_1, std_1)
    #     #     #self.ant[a]['high'] = fz.trimf(self.ant[a].universe, [min_1, peak_1, max_1])
    #     #     self.ant[a]['high'] = fz.trapmf(self.ant[a].universe, [s['min1'], s['q251'], s['q751'], s['max1']])
    #     #     self.ant[a].view()

    def feature_std(self, feat, df, target):
        return df.loc[df['Diagnosis'] == target, feat].std()

    def feature_mean(self, feat, df, target):
        return df.loc[df['Diagnosis'] == target, feat].mean()

    def feature_min(self, feat, df, target):
        return df.loc[df['Diagnosis'] == target, feat].min()

    def feature_max(self, feat, df, target):
        return df.loc[df['Diagnosis'] == target, feat].max()

    def feature_quantile(self, feat, df, target, q):
        return df.loc[df['Diagnosis'] == target, feat].quantile(q)

    def feature_kde_peak(self, feat, df, target):
        kernel = stats.gaussian_kde(df.loc[df['Diagnosis'] == target, feat])
        universe = np.linspace(df[feat].min(), df[feat].max(), num=200)
        kernel = kernel(universe)
        return universe[np.argsort(kernel)[-1]]

    def mf_gaussmf(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.gaussmf(self.ant[a].universe, s['mean' + t], s['std' + t])

    def transform_class_to_target(self, t):
        return 0 if t == 'low' else 1

    def mf_trimf(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.trimf(self.ant[a].universe, [s['min' + t], s['pke' + t], s['max' + t]])

    def mf_trapfm(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.trapmf(self.ant[a].universe, [s['min' + t], s['q25' + t], s['q75' + t], s['max' + t]])

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

    def logistic_regression_model(self):
        lr = LogisticRegression(penalty='l2', solver='sag', max_iter=self.max_iters)
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)
        self.accuracy_score = accuracy_score(self.y_test, y_pred)
        print('Accuracy ', self.accuracy_score)

    def decision_tree_model(self):
        dtc = DecisionTreeClassifier(random_state=self.random_state)
        dtc.fit(self.X_train, self.y_train)

    def random_forect_model(self):
        rfc = RandomForestClassifier(max_depth=2, random_state=self.random_state)
        rfc.fit(self.X_train, self.y_train)


wdbcFis = WDBCFis()
