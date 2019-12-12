import time
from contextlib import contextmanager
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, RFE, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler



@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class WisconsinBreastCancer:
    def __init__(self):
        self.random_state = 20
        self.column_stats = {}
        self.wdbc_full = None
        self.X = None
        self.y = None
        self.X_scaled = None

        self.label_map_string_2_int = {'M': 0, 'B': 1}

        self.features_selected = []
        self.highly_correlated_features = []

        with timer('\nLoad dataset'):
            self.load_data()
            self.encode_target()

        with timer('\nInitial Statistics'):
            self.column_statistics()
            self.row_count_by_target('Diagnosis')

        with timer('\nSetting X and y'):
            self.set_X_y()

        with timer('\nFeature Selection'):
            self.univariate_feature_selection()
            self.random_forest_classifier()

        with timer('\nCondensing X to selected features'):
            self.X = self.X[self.features_selected]

        with timer('\nDistributions'):
            self.correlation_heatmap()
            self.update_X_drop_highly_corr()
            self.scale()
            self.distribution_multi_kde()
            self.distplot()
            self.boxplot()

    def load_data(self):
        self.wdbc_full = pd.read_csv('data/wdbc.data', header=None)

        self.wdbc_full.columns = ['ID', 'Diagnosis', 'RadiusMean', 'TextureMean', 'PerimeterMean', 'AreaMean',
                                  'SmoothnessMean', 'CompactnessMean', 'ConcavityMean', 'ConcavePointsMean',
                                  'SymmetryMean', 'FractalDimensionMean', 'RadiusSe', 'TextureSe', 'PerimeterSe',
                                  'AreaSe', 'SmoothnessSe', 'CompactnessSe', 'ConcavitySe', 'ConcavePointsSe',
                                  'SymmetrySe', 'FractalDimensionSe', 'RadiusMax', 'TextureMax', 'PerimeterMax',
                                  'AreaMax', 'SmoothnessMax', 'CompactnessMax', 'ConcavityMax', 'ConcavePointsMax',
                                  'SymmetryMax', 'FractalDimensionMax']

        print('\n--- Shape after data load')
        self.print_shape()

    def encode_target(self):
        self.wdbc_full['Diagnosis'] = self.wdbc_full['Diagnosis'].map(self.label_map_string_2_int)
        self.wdbc_full['Diagnosis'].astype(int)

    def column_statistics(self):
        print('\n--- Column Stats')
        for col in self.wdbc_full:
            self.column_stats[col + '_dtype'] = self.wdbc_full[col].dtype
            self.column_stats[col + '_zero_num'] = (self.wdbc_full[col] == 0).sum()
            self.column_stats[col + '_zero_num'] = self.column_stats[col + '_zero_num'] + (self.wdbc_full[col] == '?').sum()
            self.column_stats[col + '_zero_pct'] = (((self.wdbc_full[col] == 0).sum() / self.wdbc_full.shape[0]) * 100)
            self.column_stats[col + '_nunique'] = self.wdbc_full[col].nunique()

            print('\n- {} ({})'.format(col, self.column_stats[col + '_dtype']))
            print('\tzero {} ({:.2f}%)'.format(self.column_stats[col + '_zero_num'],
                                               self.column_stats[col + '_zero_pct']))
            print('\tdistinct {}'.format(self.column_stats[col + '_nunique']))

            # Numerical features
            if self.wdbc_full[col].dtype != object:
                self.column_stats[col + '_min'] = self.wdbc_full[col].min()
                self.column_stats[col + '_mean'] = self.wdbc_full[col].mean()
                self.column_stats[col + '_quantile_25'] = self.wdbc_full[col].quantile(.25)
                self.column_stats[col + '_quantile_50'] = self.wdbc_full[col].quantile(.50)
                self.column_stats[col + '_quantile_75'] = self.wdbc_full[col].quantile(.75)
                self.column_stats[col + '_max'] = self.wdbc_full[col].max()
                self.column_stats[col + '_std'] = self.wdbc_full[col].std()
                self.column_stats[col + '_skew'] = self.wdbc_full[col].skew()
                self.column_stats[col + '_kurt'] = self.wdbc_full[col].kurt()
                print('\tmin {}'.format(self.column_stats[col + '_min']))
                print('\tmean {:.3f}'.format(self.column_stats[col + '_mean']))
                print('\t25% {:.3f}'.format(self.column_stats[col + '_quantile_25']))
                print('\t50% {:.3f}'.format(self.column_stats[col + '_quantile_50']))
                print('\t75% {:.3f}'.format(self.column_stats[col + '_quantile_75']))
                print('\tmax {}'.format(self.column_stats[col + '_max']))
                print('\tstd {:.3f}'.format(self.column_stats[col + '_std']))
                print('\tskew {:.3f}'.format(self.column_stats[col + '_skew']))
                print('\tkurt {:.3f}'.format(self.column_stats[col + '_kurt']))

    def set_X_y(self):
        self.X = self.wdbc_full.copy()
        self.X.drop(['ID', 'Diagnosis'], axis=1, inplace=True)
        self.y = pd.Series(self.wdbc_full.Diagnosis)

    def univariate_feature_selection(self):
        k = 15
        print('\n', '_' * 40, 'Univariate feature selection with chi-squared', '_' * 40)
        kbest = SelectKBest(score_func=chi2, k=15)
        fit = kbest.fit(self.X, self.y)
        print(fit.scores_)
        cols = kbest.get_support()
        features_selected = self.X.columns[cols]
        print(features_selected)

    def random_forest_classifier(self):
        data = pd.DataFrame(columns=['Feature', 'Random Forest Importance Score'])
        k = 15
        print('\n\n', '_' * 40, 'Random Forest Classifier', '_' * 40)
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        model.fit(self.X, self.y)
        ranked = list(zip(self.X.columns, model.feature_importances_))
        ranked_sorted = sorted(ranked, key=lambda x: x[1], reverse=True)
        print(ranked_sorted)

        for f in ranked_sorted[:k]:
            self.features_selected.append(f[0])
            data = data.append({'Feature': f[0], 'Random Forest Importance Score': f[1]},
                               ignore_index=True)

        fig, ax = plt.subplots(figsize=(25, 7))
        ax = sns.barplot(x='Feature', y='Random Forest Importance Score', data=data, color='palevioletred')
        plt.savefig(fname='plots/wdbc feature selection RF.png', dpi=300, format='png')
        plt.show()

    def scale(self):
        scaler = MinMaxScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=self.X.columns)

    def row_count_by_target(self, target):
        print('\n--- Row count by {}'.format(target))
        series = self.wdbc_full[target].value_counts()
        for idx, val in series.iteritems():
            print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / self.wdbc_full.shape[0]) * 100)))

    def print_shape(self):
        print('\tRow count:\t', '{}'.format(self.wdbc_full.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.wdbc_full.shape[1]))

    def distribution_multi_kde(self):
        for col in self.X_scaled:
          sns.kdeplot(self.X_scaled[col], shade=True)

        plt.savefig(fname='plots/wbc distplot.png', dpi=300, format='png')
        plt.show()

    def correlation_heatmap(self, title='Correlation Heatmap', drop=False):
        # Top x selected features
        df_corr = self.wdbc_full[self.features_selected].copy()
        corr = df_corr.corr()
        plt.clf()
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_title(title, size=16)
        drop_self = np.zeros_like(corr)  # Drop self-correlations
        drop_self[np.triu_indices_from(drop_self)] = True
        g = sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", mask=drop_self, cbar=True)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        g.set_xticklabels(g.get_yticklabels(), rotation=45)
        plt.savefig(fname='plots/wdbc corr heatmap top features.png', dpi=300, format='png')
        plt.show()

        # Drop highly correlated for 2nd heatmap
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        self.highly_correlated_features = [column for column in upper.columns if any(upper[column] > 0.95)]
        df_corr = df_corr.drop(df_corr[self.highly_correlated_features], axis=1)
        corr = df_corr.corr()
        plt.clf()
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_title(title, size=16)
        drop_self = np.zeros_like(corr)  # Drop self-correlations
        drop_self[np.triu_indices_from(drop_self)] = True
        g = sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", mask=drop_self,
                        cbar=True)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        g.set_xticklabels(g.get_yticklabels(), rotation=45)
        plt.savefig(fname='plots/wdbc corr heatmap top features after dropping highly correlated.png', dpi=300, format='png')
        plt.show()

    def update_X_drop_highly_corr(self):
        self.X = self.X.drop(self.X[self.highly_correlated_features], axis=1)

    def distplot(self):
        for col in self.X:
            plt.ylabel('Density')
            sns.distplot(self.X[col], hist=True, color='palevioletred', kde=True, hist_kws={'edgecolor':'black'},
                         kde_kws={'linewidth': 4})
            plt.savefig(fname='plots/wdbc dist ' + col + '.png', dpi=300, format='png')
            plt.show()

    def boxplot(self):
        palette = {1: "g", 0: "r"}
        for col in self.X:
            sns.catplot(x='Diagnosis', y=col, kind='violin', data=self.wdbc_full, palette=palette)
            plt.savefig(fname='plots/wdbc violin ' + col + '.png', dpi=300, format='png')
            plt.show()


wisconsinBreastCancer = WisconsinBreastCancer()
