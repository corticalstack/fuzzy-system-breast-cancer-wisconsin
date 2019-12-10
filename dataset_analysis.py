import time
from contextlib import contextmanager
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class WisconsinBreastCancer:
    def __init__(self):
        self.random_state = 20
        self.column_stats = {}
        self.wbc_test = None
        self.wbc_train = None
        self.wbc_full = None

        self.label_map_int_2_string = {2: 'benign', 4: 'malignant'}
        self.label_map_string_2_int = {'benign': 2, 'malignant': 4}

        with timer('\nPreparing dataset'):
            self.load_data()

        with timer('\nStatistics'):
            self.drop_rows_with_nulls()
            self.drop_duplicate_sample_code_numbers()
            self.set_columns_to_numeric()
            self.column_statistics()
            self.row_count_by_target('target')
            self.decode_target()
            #self.distribution()
            #self.correlation_heatmap()
            #self.boxplot()
            self.chi_squared()

    def load_data(self):
        self.wbc_full = pd.read_csv('data/breast-cancer-wisconsin.data', header=None)

        self.wbc_full.columns = ['sample_code_number', 'clump_thickness', 'uniformity_cell_size',
                                 'uniformity_cell_shape', 'marginal_adhesion', 'single_epithelial_cell_size',
                                 'number_bare_nuclei', 'bland_chromatin', 'number_normal_nucleoli', 'mitoses',
                                 'target']
        print('\n--- Shape after data load')
        self.print_shape()

    def drop_rows_with_nulls(self):
        self.wbc_full.replace('?', np.nan, inplace=True)
        self.wbc_full.dropna(axis=0, inplace=True)
        print('\n--- Shape after dropping rows with nulls')
        self.print_shape()

    def drop_duplicate_sample_code_numbers(self):
        self.wbc_full.drop_duplicates('sample_code_number', inplace=True)
        self.wbc_full.reset_index(inplace=True, drop=True)
        print('\n--- Shape after dropping duplicate sample code numbers')
        self.print_shape()

    def set_columns_to_numeric(self):
        self.wbc_full['number_bare_nuclei'] = pd.to_numeric(self.wbc_full['number_bare_nuclei'])

    def column_statistics(self):
        print('\n--- Column Stats')
        for col in self.wbc_full:
            self.column_stats[col + '_dtype'] = self.wbc_full[col].dtype
            self.column_stats[col + '_zero_num'] = (self.wbc_full[col] == 0).sum()
            self.column_stats[col + '_zero_num'] = self.column_stats[col + '_zero_num'] + (self.wbc_full[col] == '?').sum()
            self.column_stats[col + '_zero_pct'] = (((self.wbc_full[col] == 0).sum() / self.wbc_full.shape[0]) * 100)
            self.column_stats[col + '_nunique'] = self.wbc_full[col].nunique()

            print('\n- {} ({})'.format(col, self.column_stats[col + '_dtype']))
            print('\tzero {} ({:.2f}%)'.format(self.column_stats[col + '_zero_num'],
                                               self.column_stats[col + '_zero_pct']))
            print('\tdistinct {}'.format(self.column_stats[col + '_nunique']))

            # Numerical features
            if self.wbc_full[col].dtype != object:
                self.column_stats[col + '_min'] = self.wbc_full[col].min()
                self.column_stats[col + '_mean'] = self.wbc_full[col].mean()
                self.column_stats[col + '_quantile_25'] = self.wbc_full[col].quantile(.25)
                self.column_stats[col + '_quantile_50'] = self.wbc_full[col].quantile(.50)
                self.column_stats[col + '_quantile_75'] = self.wbc_full[col].quantile(.75)
                self.column_stats[col + '_max'] = self.wbc_full[col].max()
                self.column_stats[col + '_std'] = self.wbc_full[col].std()
                self.column_stats[col + '_skew'] = self.wbc_full[col].skew()
                self.column_stats[col + '_kurt'] = self.wbc_full[col].kurt()
                print('\tmin {}'.format(self.column_stats[col + '_min']))
                print('\tmean {:.3f}'.format(self.column_stats[col + '_mean']))
                print('\t25% {:.3f}'.format(self.column_stats[col + '_quantile_25']))
                print('\t50% {:.3f}'.format(self.column_stats[col + '_quantile_50']))
                print('\t75% {:.3f}'.format(self.column_stats[col + '_quantile_75']))
                print('\tmax {}'.format(self.column_stats[col + '_max']))
                print('\tstd {:.3f}'.format(self.column_stats[col + '_std']))
                print('\tskew {:.3f}'.format(self.column_stats[col + '_skew']))
                print('\tkurt {:.3f}'.format(self.column_stats[col + '_kurt']))

    def row_count_by_target(self, target):
        print('\n--- Row count by {}'.format(target))
        series = self.wbc_full[target].value_counts()
        for idx, val in series.iteritems():
            print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / self.wbc_full.shape[0]) * 100)))

    def print_shape(self):
        print('\tRow count:\t', '{}'.format(self.wbc_full.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.wbc_full.shape[1]))

    def decode_target(self):
        self.wbc_full['target'].astype(str)
        self.wbc_full['target'] = self.wbc_full['target'].map(self.label_map_int_2_string)

    def distribution(self):
        for col in self.wbc_full:
            if col != 'target' and col != 'sample_code_number':
                sns.kdeplot(self.wbc_full[col], shade=True)

        plt.savefig(fname='plots/wbc distplot.png', dpi=300, format='png')
        plt.show()

    def correlation_heatmap(self, title='Correlation Heatmap', drop=False):
        df_corr = self.wbc_full[['clump_thickness', 'uniformity_cell_size', 'uniformity_cell_shape',
                                 'marginal_adhesion', 'single_epithelial_cell_size', 'number_bare_nuclei',
                                 'bland_chromatin', 'number_normal_nucleoli', 'mitoses']].copy()
        corr = df_corr.corr()
        plt.clf()
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_title(title, size=16)
        drop_self = np.zeros_like(corr)  # Drop self-correlations
        drop_self[np.triu_indices_from(drop_self)] = True
        g = sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", mask=dropSelf, cbar=True)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        g.set_xticklabels(g.get_yticklabels(), rotation=45)
        plt.savefig(fname='plots/corr heatmap.png', dpi=300, format='png')
        plt.show()

    def boxplot(self):
        palette = {"benign": "g", "malignant": "r"}

        sns.catplot(x='target', y='clump_thickness', kind='violin', data=self.wbc_full, palette=palette)
        plt.savefig(fname='plots/wbc clump_thickness violin.png', dpi=300, format='png')
        plt.show()

        sns.catplot(x='target', y='uniformity_cell_size', kind='violin', data=self.wbc_full, palette=palette)
        plt.savefig(fname='plots/wbc uniformity_cell_size.png', dpi=300, format='png')
        plt.show()

        sns.catplot(x='target', y='uniformity_cell_shape', kind='violin', data=self.wbc_full, palette=palette)
        plt.savefig(fname='plots/wbc uniformity_cell_shape.png', dpi=300, format='png')
        plt.show()

        sns.catplot(x='target', y='marginal_adhesion', kind='violin', data=self.wbc_full, palette=palette)
        plt.savefig(fname='plots/wbc marginal_adhesion.png', dpi=300, format='png')
        plt.show()

        sns.catplot(x='target', y='single_epithelial_cell_size', kind='violin', data=self.wbc_full, palette=palette)
        plt.savefig(fname='plots/wbc single_epithelial_cell_size.png', dpi=300, format='png')
        plt.show()

        sns.catplot(x='target', y='number_bare_nuclei', kind='violin', data=self.wbc_full, palette=palette)
        plt.savefig(fname='plots/wbc number_bare_nuclei.png', dpi=300, format='png')
        plt.show()

        sns.catplot(x='target', y='bland_chromatin', kind='violin', data=self.wbc_full, palette=palette)
        plt.savefig(fname='plots/wbc bland_chromatin.png', dpi=300, format='png')
        plt.show()

        sns.catplot(x='target', y='number_normal_nucleoli', kind='violin', data=self.wbc_full, palette=palette)
        plt.savefig(fname='plots/wbc number_normal_nucleoli.png', dpi=300, format='png')
        plt.show()

        sns.catplot(x='target', y='mitoses', kind='violin', data=self.wbc_full, palette=palette)
        plt.savefig(fname='plots/wbc mitoses.png', dpi=300, format='png')
        plt.show()

    def chi_squared(self):
        k = 9
        X = self.wbc_full.drop(['target', 'sample_code_number'], axis=1)
        y = self.wbc_full['target']

        k_best_features = SelectKBest(score_func=chi2, k=k)
        fit = k_best_features.fit(X, y)
        k_best_i = np.argsort(k_best_features.scores_)[::-1]
        data = pd.DataFrame(columns=['Feature', 'Chi-Squared Score'])

        features = []
        for i in range(k):
            data = data.append({'Feature': X.columns[k_best_i[i]], 'Chi-Squared Score': k_best_features.scores_[k_best_i[i]]},
                               ignore_index=True)
            features.append(X.columns[k_best_i[i]])

        fig, ax = plt.subplots(figsize=(20, 7))
        ax = sns.barplot(x='Feature', y='Chi-Squared Score', data=data, color='palevioletred')
        plt.savefig(fname='plots/wbc feature selection chi squared.png', dpi=300, format='png')
        plt.show()


wisconsinBreastCancer = WisconsinBreastCancer()
