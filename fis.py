# Seed the NumPy random generator
from numpy.random import seed
seed(1)

import tensorflow as tf
#tf.random.set_seed(2)

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
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers

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

        self.predict_log = []

        # Globals for test enabled and test config
        self.t_enabled = None
        self.t_id = None
        self.t_rs = None
        self.t_mfgrp = None

        # Support running non-fis models only once per feature rule set
        self.last_rs = None

        # Network parameters
        self.epochs = 40
        self.batch_size = 100
        self.verbose = 0

        # FIS components
        self.ant = {}
        self.diagnosis = None
        self.rules = []
        self.ops = {
            '&': operator.and_,
            '|': operator.or_
        }
        self.system = None
        self.crisp_binary_threshold = 0

        # Map cycling through range to defuzzification methods to actual mode name
        self.defuzzify_switcher = {1: 'centroid', 2: 'bisector', 3: 'mom', 4: 'som', 5: 'lom'}

        # Support plotting of unique MF only once
        self.mf_plotted = {}

        # Suite of FIS tests that include antecedents and their configured term MFs, consequent terms and Mfs,
        # and rules linking the 2
        self.tests = [[{'Config': [{'Id': 1, 'Rs': 1, 'Mfgrp': 'gauss', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                                  {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', '-op': '&',
                                   'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 2, 'Rs': 1, 'Mfgrp': 'zs', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'zmf', 'high': 'smf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                                  {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', '-op': '&',
                                   'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 3, 'Rs': 1, 'Mfgrp': 'mix', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'trimf', 'high': 'trapmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'trimf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                                  {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', '-op': '&',
                                   'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 4, 'Rs': 2, 'Mfgrp': 'gauss', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'AreaSe': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'TextureMean': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'SmoothnessMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', 'TextureMean':
                           'low', 'SmoothnessMax': 'low', '-op': '&', 'Diagnosis': 'benign'}, {'PerimeterMax': 'high',
                           'ConcavePointsMax': 'high', 'AreaSe': 'high', 'TextureMean': 'high', 'SmoothnessMax': 'high',
                           '-op': '&', 'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 5, 'Rs': 2, 'Mfgrp': 'zs', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'AreaSe': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'TextureMean': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'SmoothnessMax': {'mf': {'low': 'zmf', 'high': 'smf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [
                           {'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', 'TextureMean': 'low',
                            'SmoothnessMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                           {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'AreaSe': 'high', 'TextureMean': 'high',
                            'SmoothnessMax': 'high', '-op': '&', 'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 6, 'Rs': 2, 'Mfgrp': 'mix', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'trimf', 'high': 'trapmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'trimf', 'high': 'gaussmf'}}},
                                {'AreaSe': {'mf': {'low': 'trimf', 'high': 'gaussmf'}}},
                                {'TextureMean': {'mf': {'low': 'trimf', 'high': 'trimf'}}},
                                {'SmoothnessMax': {'mf': {'low': 'trimf', 'high': 'trimf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [
                           {'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', 'TextureMean': 'low',
                            'SmoothnessMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                           {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'AreaSe': 'high', 'TextureMean': 'high',
                            'SmoothnessMax': 'high', '-op': '&', 'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 7, 'Rs': 3, 'Mfgrp': 'gauss', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'AreaSe': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', '-op': '&',
                                   'Diagnosis': 'benign'},
                                  {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'AreaSe': 'high', '-op': '&',
                                   'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 8, 'Rs': 3, 'Mfgrp': 'zs', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'AreaSe': {'mf': {'low': 'zmf', 'high': 'smf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', '-op': '&',
                                   'Diagnosis': 'benign'},
                                  {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'AreaSe': 'high', '-op': '&',
                                   'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 9, 'Rs': 3, 'Mfgrp': 'mix', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'trimf', 'high': 'trapmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'trimf', 'high': 'gaussmf'}}},
                                {'AreaSe': {'mf': {'low': 'trimf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', '-op': '&',
                                   'Diagnosis': 'benign'},
                                  {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'AreaSe': 'high', '-op': '&',
                                   'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 10, 'Rs': 4, 'Mfgrp': 'gauss', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'AreaSe': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'TextureMean': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', 'TextureMean':
                           'low', '-op': '&', 'Diagnosis': 'benign'},
                                  {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'AreaSe': 'high', 'TextureMean':
                                      'high', '-op': '&', 'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 11, 'Rs': 4, 'Mfgrp': 'zs', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'AreaSe': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'TextureMean': {'mf': {'low': 'zmf', 'high': 'smf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', 'TextureMean':
                           'low', '-op': '&', 'Diagnosis': 'benign'},
                                  {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'AreaSe': 'high', 'TextureMean':
                                      'high', '-op': '&', 'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 12, 'Rs': 4, 'Mfgrp': 'mix', 'Enabled': True}]},
                       {'Ant': [{'PerimeterMax': {'mf': {'low': 'trimf', 'high': 'trapmf'}}},
                                {'ConcavePointsMax': {'mf': {'low': 'trimf', 'high': 'gaussmf'}}},
                                {'AreaSe': {'mf': {'low': 'trimf', 'high': 'gaussmf'}}},
                                {'TextureMean': {'mf': {'low': 'trimf', 'high': 'trimf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'PerimeterMax': 'low', 'ConcavePointsMax': 'low', 'AreaSe': 'low', 'TextureMean':
                           'low', '-op': '&', 'Diagnosis': 'benign'},
                                  {'PerimeterMax': 'high', 'ConcavePointsMax': 'high', 'AreaSe': 'high', 'TextureMean':
                                      'high', '-op': '&', 'Diagnosis': 'malignant'}]}],

                      [{'Config': [{'Id': 13, 'Rs': 5, 'Mfgrp': 'gauss', 'Enabled': True}]},
                       {'Ant': [{'TextureMean': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}},
                                {'SmoothnessMax': {'mf': {'low': 'gaussmf', 'high': 'gaussmf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                       {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'TextureMean': 'low', 'SmoothnessMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                                  {'TextureMean': 'high', 'SmoothnessMax': 'high', '-op': '&', 'Diagnosis':
                                      'malignant'}]}],

                      [{'Config': [{'Id': 14, 'Rs': 5, 'Mfgrp': 'zs', 'Enabled': True}]},
                       {'Ant': [{'TextureMean': {'mf': {'low': 'zmf', 'high': 'smf'}}},
                                {'SmoothnessMax': {'mf': {'low': 'zmf', 'high': 'smf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'TextureMean': 'low', 'SmoothnessMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                                  {'TextureMean': 'high', 'SmoothnessMax': 'high', '-op': '&', 'Diagnosis':
                                      'malignant'}]}],

                      [{'Config': [{'Id': 15, 'Rs': 5, 'Mfgrp': 'mix', 'Enabled': True}]},
                       {'Ant': [{'TextureMean': {'mf': {'low': 'trimf', 'high': 'trimf'}}},
                                {'SmoothnessMax': {'mf': {'low': 'trimf', 'high': 'trimf'}}}]},
                       {'Con': [{'Diagnosis': {'mf': [{'benign': ['zmf', [10, 170]]},
                                                      {'malignant': ['smf', [100, 200]]}]}}]},
                       {'Rules': [{'TextureMean': 'low', 'SmoothnessMax': 'low', '-op': '&', 'Diagnosis': 'benign'},
                                  {'TextureMean': 'high', 'SmoothnessMax': 'high', '-op': '&', 'Diagnosis':
                                      'malignant'}]}]
                      ]

        with timer('\nLoad dataset'):
            self.load_data()

        with timer('\nSplit dataset'):
            self.set_y()
            self.remove_target_from_X()

        with timer('\nTesting Scenarios'):
            for i, t in enumerate(self.tests):
                self.set_test_global_cfg(t)

                if not self.t_enabled:
                    continue

                with timer('\nTest ' + str(self.t_id) + '_' + str(self.t_rs)):

                    # Clear any previous FIS test config
                    self.ant_cfg = []
                    self.con_cfg = []
                    self.rule_cfg = []

                    self.set_test_ant_con_cfg(t)

                with timer('\nPreparing dataset for test'):

                    # Partition dataset and set partition cols as per FIS test antecedents
                    self.train_test_split()
                    self.set_X_train_test_cols()
                    self.set_X_y_train()

                # Execute non-FIS model predict and score only once per RS
                if self.last_rs != self.t_rs:
                    self.last_rs = self.t_rs
                    with timer('\nLogistic Regression ML Model Prediction'):
                        self.lr_model_predict_score()
                    with timer('\nDecision Tree ML Model Prediction'):
                        self.dtc_model_predict_score()
                    with timer('\nRandom Forest Classification ML Model Prediction'):
                        self.rfc_model_predict_score()
                    with timer('\nNeural Network ML Model Prediction'):
                        self.nn_model_predict_score()


                # For each defuzzification method
                for d in range(1, 6):
                    self.ant = {}
                    self.diagnosis = None
                    self.rules = []

                    # Prepare FIS components
                    self.create_antecendents_universe(defuzzify_method=self.defuzzify_switcher[d])
                    self.create_consequent_universe(defuzzify_method=self.defuzzify_switcher[d])
                    self.set_antecendents_mfs()
                    self.set_consequent_mfs()
                    self.set_rules(t)

                    self.system = ct.ControlSystem(rules=self.rules)
                    self.diagnose = ct.ControlSystemSimulation(self.system)

                    # For each crisp to binary threshold limit in required range
                    for c2b in range(90, 131):
                        with timer('\nMaking FIS Model Predictions with defuzzify method ' + self.defuzzify_switcher[d]
                                   + ' & threshold ' + str(c2b)):
                            # Compute FIS and score classification
                            self.fis_model_predict_score(self.defuzzify_switcher[d], c2b)

            self.download_predict_log()

    # Suite of methods for feature distribution statistics that support MF shaping
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

    @staticmethod
    def specificity_score(test, pred):
        tn, fp, fn, tp = confusion_matrix(test, pred).ravel()
        return tn / (tn + fp)

    def load_data(self):
        self.X = pd.read_csv('data/wdbc_selected_cols.csv')

        print('\n', '_' * 40, 'Shape After Data Load', '_' * 40)
        self.print_shape(self.X)

    def set_y(self):
        self.y = self.X['Diagnosis']

    def remove_target_from_X(self):
        self.X.drop('Diagnosis', axis=1, inplace=True)

    # Split dataset into train (70%) and test (30%) partitions
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

    def set_test_global_cfg(self, t):
        self.t_enabled = t[0]['Config'][0]['Enabled']
        self.t_id = t[0]['Config'][0]['Id']
        self.t_rs = t[0]['Config'][0]['Rs']
        self.t_mfgrp = t[0]['Config'][0]['Mfgrp']

    # Parse test config grabbing MF term and shape
    def set_test_ant_con_cfg(self, t):
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
                    # Execute MF function associated with configured antecedent
                    self.ant[a][v[0]] = getattr(self, 'mf_' + v[1])(a, v[0], s)
        if a + v[1] not in self.mf_plotted: # If not done then plot term MFs for antecedent
            self.mf_plotted[a + v[1]] = True
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

    # Suite of calls to skfuzzy functions to shape term MFs to feature distributions
    def mf_gaussmf(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.gaussmf(self.ant[a].universe, s['mean' + t], s['std' + t])

    def mf_trimf(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.trimf(self.ant[a].universe, [s['min' + t], s['pke' + t], s['max' + t]])

    def mf_trapmf(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.trapmf(self.ant[a].universe, [s['min' + t], s['q25' + t], s['q75' + t], s['max' + t]])

    def mf_zmf(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.zmf(self.ant[a].universe, s['min' + t], s['pke' + t])

    def mf_smf(self, a, t, s):
        t = str(self.transform_class_to_target(t))
        return fz.smf(self.ant[a].universe, s['min' + t], s['pke' + t])

    def set_consequent_mfs(self):
        for c in self.con_cfg:
            for k, v in c.items():
                self.diagnosis[v[0]] = getattr(fz, v[1][0])(self.diagnosis.universe, v[1][1][0], v[1][1][1])

        if 'diagnosis' not in self.mf_plotted:
            self.mf_plotted['diagnosis'] = True
            self.diagnosis.view()

    def set_rules(self, test):
        for r in test:
            if 'Rules' in r:
                self.add_rules(r['Rules'])

    # Parse configured test rules
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
                        antecedent = op_func(antecedent, self.ant[arg[0]][arg[1]])  # Link antecedent terms
            r = ct.Rule(antecedent, consequent=consequent, label=label)
            self.rules.append(r)

    def fis_model_predict_score(self, defuzzify_method, crisp_threshold):
        y_pred = []

        # Set to true to support debugging and output analysis
        dv = False
        ychk = False

        # For each sample in test partition
        for di, dr in self.X_test.iterrows():
            # Take each feature value in test sample and pass it to previously configured antecedent
            for si, sv in dr.iteritems():
                self.diagnose.input[si] = sv
            try:
                self.diagnose.compute()  # Now execute the FIS
                #self.diagnose.print_state()
                if dv:
                    self.diagnosis.view(sim=self.diagnose)
            except ValueError:
                print(self.diagnose.input)
                continue

            # Transform crsip to binary using threshold
            crisp_to_binary = 0 if self.diagnose.output['diagnosis'] < crisp_threshold else 1
            if ychk:  # If set to true, supports checking terms MF shape
                if self.y_test.loc[di] != crisp_to_binary:
                    print('Ground truth is ', self.y_test.loc[di], ' and pred is ', crisp_to_binary)
                    print('Crisp output is ', crisp_to_binary)
                    print('Inputs are')
                    for si, sv in dr.iteritems():
                        print(si, ' = ', sv)

                    self.diagnosis.view(sim=self.diagnose)

            # Add to array of binary predictions for test set
            y_pred.append(crisp_to_binary)

        # Log test prediction results
        ref = 'rs' + str(self.t_rs) + '_id' + str(self.t_id) + '_FIS_ct' + str(crisp_threshold) + '_' + defuzzify_method
        self.log_prediction(ref, self.t_rs, self.t_id, self.t_mfgrp, 'FIS', defuzzify_method, crisp_threshold,
                            self.y_test, y_pred)

        # Plot confusion matrix
        self.plot_cm(self.y_test, y_pred, ref)

    def lr_model_predict_score(self):
        lr = LogisticRegression(random_state=self.random_state)
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)
        ref = 'rs' + str(self.t_rs) + '_LR'
        self.log_prediction(ref, self.t_rs, 'n/a', 'n/a', 'LR', 'n/a', 'n/a', self.y_test, y_pred)
        self.plot_cm(self.y_test, y_pred, ref)

    def dtc_model_predict_score(self):
        dtc = DecisionTreeClassifier(random_state=self.random_state)
        dtc.fit(self.X_train, self.y_train)
        y_pred = dtc.predict(self.X_test)
        ref = 'rs' + str(self.t_rs) + '_DTC'
        self.log_prediction(ref, self.t_rs, 'n/a', 'n/a', 'DTC', 'n/a', 'n/a', self.y_test, y_pred)
        self.plot_cm(self.y_test, y_pred, ref)

    def rfc_model_predict_score(self):
        rfc = RandomForestClassifier(random_state=self.random_state)
        rfc.fit(self.X_train, self.y_train)
        y_pred = rfc.predict(self.X_test)
        ref = 'rs' + str(self.t_rs) + '_RFC'
        self.log_prediction(ref, self.t_rs, 'n/a', 'n/a', 'RFC', 'n/a', 'n/a', self.y_test, y_pred)
        self.plot_cm(self.y_test, y_pred, ref)

    def nn_model_predict_score(self):
        # Build model
        model = models.Sequential()
        model.add(layers.Dense(self.X_train.shape[1] * 4, kernel_initializer='normal', activation='relu',
                               input_shape=(self.X_train.shape[1],)))
        model.add(layers.Dense(self.X_train.shape[1] * 4, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Keras requires numpy arrays (not dataframes)
        X_train = self.X_train.to_numpy()
        y_train = self.y_train.to_numpy()
        X_test = self.X_test.to_numpy()
        y_test = self.y_test.to_numpy()

        self.history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs, verbose=self.verbose, batch_size = 1)

        y_pred = model.predict_classes(X_test)

        ref = 'rs' + str(self.t_rs) + '_NN'
        self.log_prediction(ref, self.t_rs, 'n/a', 'n/a', 'NN', 'n/a', 'n/a', self.y_test, y_pred)
        self.plot_cm(self.y_test, y_pred, ref)

    def plot_cm(self, y, y_pred, ref):
        cm = confusion_matrix(y, y_pred)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, annot_kws={"size": 14}, fmt='d', cmap='Greens', cbar=False)
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix - ' + ref)
        ax.xaxis.set_ticklabels(['Benign', 'Malignant'])
        ax.yaxis.set_ticklabels(['Benign', 'Malignant'])
        plt.savefig(fname='plots/cm/CM - ' + ref + '.png', dpi=300, format='png')
        plt.close()
        #plt.show()

    def log_prediction(self, ref, trs, tid, mfgrp, clf, dm, cbt, y_test, y_pred):
        acc = round(accuracy_score(y_test, y_pred), 3)
        sen = round(recall_score(y_test, y_pred), 3)
        spe = round(self.specificity_score(y_test, y_pred), 3)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        self.predict_log.append([ref, trs, tid, mfgrp, clf, dm, cbt, tp, fp, tn, fn, sen, acc, spe])

    # Save full and summarised prediction log to csv for analysis
    def download_predict_log(self):
        df_pred_log = pd.DataFrame(self.predict_log, columns=['Ref', 'Rs', 'Id', 'Mfgrp', 'Clf', 'Dm', 'Ct', 'Tp', 'Fp',
                                                              'Tn', 'Fn', 'Sen', 'Acc', 'Spe'])
        df_pred_log.sort_values(by=['Sen', 'Acc', 'Spe', 'Dm', 'Ct', 'Clf',  'Rs', 'Id'],
                                ascending=[False, False, False, True, True, True, True, True], inplace=True)
        df_pred_log['Rank'] = np.arange(1, len(df_pred_log) + 1)
        df_pred_log.to_csv('data/predict_log.csv', header=True, index=False)

        df_log_sum = df_pred_log[['Rank', 'Clf', 'Ref', 'Rs', 'Mfgrp', 'Dm', 'Tp', 'Fp', 'Tn', 'Fn', 'Sen', 'Acc', 'Spe']].copy()
        df_log_sum.drop_duplicates(subset=['Clf', 'Rs', 'Mfgrp', 'Dm'], keep='first', inplace=True)
        df_log_sum.to_csv('data/predict_log_sum.csv', header=True, index=False)


wdbcFis = WDBCFis()

