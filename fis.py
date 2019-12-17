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

        self.tests = [[{'Antecedents': [{'PerimeterMax': {'mf': {'low': 'gaussmf',
                                                                 'high': 'trimf'}}}]},
                       {'Consequent': [{'Diagnosis': {'mf': {'benign': {'mf': 'trapmf',
                                                                        'shape': [0, 10, 40, 50]},
                                                             'malignant': {'mf': 'trapmf',
                                                                           'shape': [50, 60, 90, 100]}}}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'Consequent': 'benign'}]}],
                      [{'Antecedents': [{'PerimeterMax': {'mf': {'low': 'gaussmf',
                                                                 'high': 'gaussmf'}}},
                                        {'ConcavePointsMax': {'mf': {'low': 'gaussmf',
                                                                     'high': 'gaussmf'}}}]},
                       {'Consequent': [{'Diagnosis': {'mf': {'benign': {'mf': 'trapmf',
                                                                        'shape': [0, 10, 40, 50]},
                                                             'malignant': {'mf': 'trapmf',
                                                                           'shape': [50, 60, 90, 100]}}}}]},
                       {'Rules': [[{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'antop': '&', 'Consequent': 'benign'}],
                                  [{'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'antop': '&', 'Consequent': 'malignant'}]]}]]

        self.ant_cfg = []
        self.con_cfg = []
        self.rule_cfg = []

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

                with timer('\nCreating Consequent Universe'):
                    self.create_consequent_universe()

                with timer('\nSetting Antecedent MFs'):
                    self.set_antecendents_mfs()

                with timer('\nSetting Consequent'):
                    self.set_consequent_mfs()

                with timer('\nSetting Rules'):
                    self.set_test_rules(t)

                with timer('\nSetting System'):
                    self.system = ct.ControlSystem(rules=self.rules)
                    self.sim = ct.ControlSystemSimulation(self.system)

                with timer('\nMaking Predictions'):
                    # Should lop through all test data here, store, defuzzify, compoare with ground truth
                    self.predict()

    @staticmethod
    def feature_std(feat, df, target):
        return df.loc[df['Diagnosis'] == target, feat].std()

    @staticmethod
    def feature_mean(feat, df, target):
        return df.loc[df['Diagnosis'] == target, feat].mean()

    @staticmethod
    def feature_min(feat, df, target):
        return df.loc[df['Diagnosis'] == target, feat].min()

    @staticmethod
    def feature_max(feat, df, target):
        return df.loc[df['Diagnosis'] == target, feat].max()

    @staticmethod
    def feature_quantile(feat, df, target, q):
        return df.loc[df['Diagnosis'] == target, feat].quantile(q)

    @staticmethod
    def feature_kde_peak(feat, df, target):
        kernel = stats.gaussian_kde(df.loc[df['Diagnosis'] == target, feat])
        universe = np.linspace(df[feat].min(), df[feat].max(), num=200)
        kernel = kernel(universe)
        return universe[np.argsort(kernel)[-1]]

    @staticmethod
    def transform_class_to_target(t):
        return 0 if t == 'low' else 1

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

    def set_test_cfg(self, t):
        for cfg in t:
            if 'Antecedents' in cfg:
                for f in cfg['Antecedents']:
                    for fk, fv in f.items():
                        if 'mf' in f[fk]:
                            for mfclass, mfshape in f[fk]['mf'].items():
                                self.ant_cfg.append({fk: (mfclass, mfshape)})
            if 'Consequent' in cfg:
                for f in cfg['Consequent']:
                    for fk, fv in f.items():
                        if 'mf' in f[fk]:
                            for mfclass, mfshape in f[fk]['mf'].items():
                                self.con_cfg.append({fk: (mfclass, mfshape)})

    def set_X_train_test_cols(self):
        cols = []
        for c in self.ant_cfg:
            for k, v in c.items():
                cols.append(k)
        cols = list(set(cols))
        self.X_train = self.X_train[cols]
        self.X_test = self.X_test[cols]

    def print_shape(self):
        print('\tRow count:\t', '{}'.format(self.X.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.X.shape[1]))

    def create_antecendents_universe(self):
        # Set universe boundary for each feature
        for feat in self.X_train:
            self.ant[feat] = ct.Antecedent(np.linspace(self.X_train[feat].min(), self.X_train[feat].max(), num=200), feat)

    def create_consequent_universe(self):
        self.diagnosis = ct.Consequent(np.arange(0, 100, 1), 'diagnosis')

    def set_antecendents_mfs(self):
        # Diagnosis: Benign = class 0, Malignant = Class 1
        for a in self.ant:
            s = self.set_antecedents_stats(a)
            self.set_antecedent_mfs(a, s)

    def set_antecedent_mfs(self, a, s):
        for c in self.ant_cfg:
            for k, v in c.items():
                if k == a:
                    self.ant[a][v[0]] = getattr(self, 'mf_' + v[1])(a, v[0], s)
        self.ant[a].view()

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

    def mf_gaussmf(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.gaussmf(self.ant[a].universe, s['mean' + t], s['std' + t])

    def mf_trimf(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.trimf(self.ant[a].universe, [s['min' + t], s['pke' + t], s['max' + t]])

    def mf_trapfm(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.trapmf(self.ant[a].universe, [s['min' + t], s['q25' + t], s['q75' + t], s['max' + t]])

    def set_consequent_mfs(self):
        for c in self.con_cfg:
            for k, v in c.items():
                self.diagnosis[v[0]] = getattr(fz, v[1]['mf'])(self.diagnosis.universe, v[1]['shape'])

        self.diagnosis.view()

    def set_test_rules(self, test):
        for r in test:
            if 'Rules' in r:
                self.add_rule(r['Rules'])

    def add_rule(self, r):
        for k, v in r:
            print(k, r[k], v)

    def set_rules_test_0(self):
        r = ct.Rule(self.ant['PerimeterMax']['low'], consequent=self.diagnosis['benign'], label='Benign')
        self.rules.append(r)

        r = ct.Rule(self.ant['PerimeterMax']['high'], consequent=self.diagnosis['malignant'], label='Malignant')
        self.rules.append(r)

    def set_rules_test_1(self):
        test = self.ant['PerimeterMax']['low']
        test = test | self.ant['ConcavePointsMax']['low']
        r = ct.Rule(test,
                      consequent=self.diagnosis['benign'], label='Benign')
        self.rules.append(r)

        r = ct.Rule(self.ant['PerimeterMax']['high'] & self.ant['ConcavePointsMax']['high'],
                      consequent=self.diagnosis['malignant'], label='Malignant')
        self.rules.append(r)

    def predict(self):
        id = 0
        sv = 0
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
