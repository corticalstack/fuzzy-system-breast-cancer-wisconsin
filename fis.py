import time
from contextlib import contextmanager
import collections
import operator
import pandas as pd
import numpy as np
import skfuzzy as fz
from skfuzzy import control as ct
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from scipy import stats
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


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

        self.X = None
        self.y = None
        self.X_train = None
        self.X_y_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_predict = []

        self.predict_log = []

        self.ant = {}
        self.diagnosis = None
        self.rules = []
        self.system = None

        self.crisp_binary_threshold = 0
        self.defuzzify_method = None
        self.defuzzify_switcher = {1: 'centroid', 2: 'bisector', 3: 'mom', 4: 'som', 5: 'lom'}

        self.ops = {
            '&': operator.and_,
            '|': operator.or_
        }

        #'CrispToBinaryThreshold': 119, 'DefuzzifyMethod': 'centroid'

        self.tests = [[{'Config': [{'Enabled': False}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'trimf', 'high': 'trapmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'trimf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                                  {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', '-op': '&',
                                   'Diagnosis': 'malignant'}]}],
                      [{'Config': [{'Enabled': False}]},
                       {'Ant': [{'AreaSe': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'SmoothnessMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [
                           {'AreaSe': 'low', 'SmoothnessMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                           {'AreaSe': 'high', 'SmoothnessMax': 'high', '-op': '&', 'Diagnosis': 'malignant'}]}],
                      [{'Config': [{'Enabled': False}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'trimf', 'high': 'trapmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'trimf', 'high': 'gaussmf'}}},
                                {'AreaSe': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'SmoothnessMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [
                           {'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', 'SmoothnessMax': 'low',
                            '-op': '&', 'Diagnosis': 'benign'},
                           {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'AreaSe': 'high', 'SmoothnessMax':
                               'high', '-op': '&', 'Diagnosis': 'malignant'}]}],
                      [{'Config': [{'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'AreaSe': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'TextureMean': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'SmoothnessMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [
                           {'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', 'TextureMean': 'low',
                            'SmoothnessMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                           {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'AreaSe': 'high', 'TextureMean': 'high',
                            'SmoothnessMax': 'high', '-op': '&', 'Diagnosis': 'malignant'}]}]
                      ]

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
                if not t[0]['Config'][0]['Enabled']:
                    continue

                with timer('\nTest ' + str(i)):
                    self.set_test_cfg(t)

                with timer('\nPreparing dataset for test '):
                    self.train_test_split()
                    self.set_X_train_test_cols()
                    self.set_X_y_train()

                    self.ant = {}
                    self.diagnosis = None
                    self.rules = []

                with timer('\nMaking Alternative ML Model Predictions'):
                    self.lr_model_predict_score(i)
                    # self.dtc_model_predict_score(i)
                    # self.rfc_model_predict_score(i)

                for d in range(1, 5):
                    with timer('\nCreating Antecedent Universe'):
                        self.create_antecendents_universe(defuzzify_method=self.defuzzify_switcher[d])

                    with timer('\nCreating Consequent Universe'):
                        self.create_consequent_universe(defuzzify_method=self.defuzzify_switcher[d])

                    with timer('\nSetting Antecedent MFs'):
                        self.set_antecendents_mfs()

                    with timer('\nSetting Consequent'):
                        self.set_consequent_mfs()

                    with timer('\nSetting Rules'):
                        self.set_rules(t)

                    with timer('\nSetting FIS System'):
                        self.system = ct.ControlSystem(rules=self.rules)
                        self.diagnose = ct.ControlSystemSimulation(self.system)

                    with timer('\nMaking Model Predictions'):
                        for c2b in range(115, 130):
                            self.fis_model_predict_score(i, self.defuzzify_switcher[d], c2b)

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
        self.print_shape(self.X)

    def set_y(self):
        self.y = self.X['Diagnosis']

    def remove_target_from_X(self):
        self.X.drop('Diagnosis', axis=1, inplace=True)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30,
                                                                                random_state=self.random_state)
        print('\n', '_' * 40, 'X_train Shape After Split', '_' * 40)
        self.print_shape(self.X_train)
        print('\n', '_' * 40, 'X_test Shape After Split', '_' * 40)
        self.print_shape(self.X_test)


    def set_X_y_train(self):
        # Used for MF shaping when feature distribution filtered by target
        self.X_y_train = pd.concat([self.X_train, self.y_train], axis=1)

    def set_test_cfg(self, t):
        for cfg in t:
            if 'Ant' in cfg:
                for f in cfg['Ant']:
                    for fk, fv in f.items():
                        if 'mf' in f[fk]:
                            for mfclass, mfshape in f[fk]['mf'].items():
                                self.ant_cfg.append({fk: (mfclass, mfshape)})
            if 'Con' in cfg:
                for f in cfg['Con']:
                    for fk, fv in f.items():
                        if 'mf' in f[fk]:
                            for terms in f[fk]['mf']:
                                for mfclass, mfshape in terms.items():
                                    self.con_cfg.append({fk: (mfclass, mfshape)})

    def set_X_train_test_cols(self):
        cols = []
        for c in self.ant_cfg:
            for k, v in c.items():
                cols.append(k)
        cols = list(set(cols))
        self.X_train = self.X_train[cols]
        self.X_test = self.X_test[cols]

    def print_shape(self, df):
        print('\tRow count:\t', '{}'.format(df.shape[0]))
        print('\tColumn count:\t', '{}'.format(df.shape[1]))

    def create_antecendents_universe(self, defuzzify_method='centroid'):
        # Set universe boundary for each feature
        for feat in self.X_train:
            self.ant[feat] = ct.Antecedent(np.linspace(self.X_train[feat].min() - (self.X_train[feat].std() * 1.1),
                                                       self.X_train[feat].max() + (self.X_train[feat].std() * 1.1),
                                                       num=200), feat, defuzzify_method=defuzzify_method)

    def create_consequent_universe(self, defuzzify_method='centroid'):
        self.diagnosis = ct.Consequent(np.arange(0, 200, 1), 'diagnosis', defuzzify_method=defuzzify_method)

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

        s['std0'] = self.feature_std(a, self.X_y_train, 0)
        s['std1'] = self.feature_std(a, self.X_y_train, 1)
        s['min0'] = self.feature_min(a, self.X_y_train, 0) - (s['std0'] * 1.1)
        s['min1'] = self.feature_min(a, self.X_y_train, 1) - (s['std1'] * 1.1)
        s['max0'] = self.feature_max(a, self.X_y_train, 0) + (s['std0'] * 1.1)
        s['max1'] = self.feature_max(a, self.X_y_train, 1) + (s['std1'] * 1.1)
        s['mean0'] = self.feature_mean(a, self.X_y_train, 0)
        s['mean1'] = self.feature_mean(a, self.X_y_train, 1)
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

    def mf_trapmf(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.trapmf(self.ant[a].universe, [s['min' + t], s['q25' + t], s['q75' + t], s['max' + t]])

    def set_consequent_mfs(self):
        for c in self.con_cfg:
            for k, v in c.items():
                self.diagnosis[v[0]] = getattr(fz, v[1][0])(self.diagnosis.universe, v[1][1][0], v[1][1][1])

        self.diagnosis.view()

    def set_rules(self, test):
        for r in test:
            if 'Rules' in r:
                self.add_rules(r['Rules'])

    def add_rules(self, rules):
        for r in rules:
            antecedent = None
            consequent = None
            label = None
            op_func = None
            od = collections.OrderedDict(sorted(r.items()))  # Required to ensure operator determined first
            for arg in od.items():
                if arg[0] == 'Diagnosis':
                    consequent = self.diagnosis[arg[1]]
                    label = arg[1]
                elif arg[0] == '-op':
                    op_func = self.ops[arg[1]]
                else:
                    if antecedent is None:
                        antecedent = self.ant[arg[0]][arg[1]]
                    else:
                        antecedent = op_func(antecedent, self.ant[arg[0]][arg[1]])
            r = ct.Rule(antecedent, consequent=consequent, label=label)
            self.rules.append(r)

    def fis_model_predict_score(self, test_num, defuzzify_method, crisp_threshold):
        test_num = str(test_num)
        y_pred = []
        for di, dr in self.X_test.iterrows():
            for si, sv in dr.iteritems():
                self.diagnose.input[si] = sv
            try:
                self.diagnose.compute()
                #self.diagnosis.view(sim=self.diagnose)
            except ValueError:
                print(self.diagnose.input)
                continue

            crisp_to_binary = 0 if self.diagnose.output['diagnosis'] < crisp_threshold else 1

            output = {'CrispOut': self.diagnose.output['diagnosis'], 'BinaryOut': crisp_to_binary}
            self.y_predict.append({'CrispOut': self.diagnose.output['diagnosis']})
            y_pred.append(output['BinaryOut'])
            #print(self.diagnose.output)
            #ants = [i for i in self.diagnose.ctrl.antecedents]
            #ant = ants[0]
            #for ant in ants:
            #    for t in ant.terms:
            #        print(ant.label, t, ant.terms[t].membership_value[self.diagnose])
            #        for t in ant.terms:
            #            print(ant.label, t, ant.terms[t].membership_value[self.diagnose])
            #print(['For term \'{0}\' membership is {1}'.format(label, term.membership_value[self.diagnose])
            #       for (label, term) in ant.terms.iteritems()])
            #self.diagnose.print_state()

        # cm = confusion_matrix(self.y_test, y_pred)
        #
        # FP = cm.sum(axis=0) - np.diag(cm)
        # FN = cm.sum(axis=1) - np.diag(cm)
        # TP = np.diag(cm)
        # TN = cm.sum() - (FP + FN + TP)
        #
        # # Sensitivity, hit rate, recall, or true positive rate
        # TPR = TP / (TP + FN)
        # # Specificity or true negative rate
        # TNR = TN / (TN + FP)
        # # Precision or positive predictive value
        # PPV = TP / (TP + FP)
        # # Negative predictive value
        # NPV = TN / (TN + FN)
        # # Fall out or false positive rate
        # FPR = FP / (FP + TN)
        # # False negative rate
        # FNR = FN / (TP + FN)
        # # False discovery rate
        # FDR = FP / (TP + FP)
        #
        # # Overall accuracy
        # ACC = (TP + TN) / (TP + FP + FN + TN)

        acc = accuracy_score(self.y_test, y_pred)
        sen = recall_score(self.y_test, y_pred)
        self.log_prediction('FIS', defuzzify_method, test_num, acc, sen)

        #print('Test ' + test_num + ' - FIS - Accuracy ' + str(acc))
        #self.confusion_matrix(self.y_test, y_pred, test_num, 'FIS')

        # JP see self.system.view_n() and see if useful, what it does
        #self.system.view_n()

    def lr_model_predict_score(self, test_num):
        test_num = str(test_num)
        lr = LogisticRegression(random_state=self.random_state)
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print('Test ' + test_num + ' - LR - Accuracy ' + str(acc))
        self.confusion_matrix(self.y_test, y_pred, test_num, 'LR')

    def dtc_model_predict_score(self, test_num):
        test_num = str(test_num)
        dtc = DecisionTreeClassifier(random_state=self.random_state)
        dtc.fit(self.X_train, self.y_train)
        y_pred = dtc.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print('Test ' + test_num + ' - DTC - Accuracy ' + str(acc))
        self.confusion_matrix(self.y_test, y_pred, test_num, 'DTC')

    def rfc_model_predict_score(self, test_num):
        test_num = str(test_num)
        rfc = RandomForestClassifier(random_state=self.random_state)
        rfc.fit(self.X_train, self.y_train)
        y_pred = rfc.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        print('Test ' + test_num + ' - RFC - Accuracy ' + str(acc))
        self.confusion_matrix(self.y_test, y_pred, test_num, 'RFC')

    def confusion_matrix(self, y, y_pred, test_num, model_name):
        cm = confusion_matrix(y, y_pred)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, annot_kws={"size": 14}, fmt='d', cmap='Greens', cbar=False)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['Benign', 'Malignant'])
        ax.yaxis.set_ticklabels(['Benign', 'Malignant'])
        plt.savefig(fname='plots/CM - test ' + test_num + ' - ' + model_name + '.png', dpi=300, format='png')
        plt.show()

    def log_prediction(self, clf, dm, test_num, acc, sen):
        self.predict_log.append([clf, dm, test_num, acc, sen])


wdbcFis = WDBCFis()
