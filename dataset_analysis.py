import time
from contextlib import contextmanager
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # Required for 3d projection
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler



@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print('{} - done in {:.0f}s'.format(title, time.time() - t0))


class WDBCAnalysis:
    def __init__(self):
        self.random_state = 20
        self.column_stats = {}
        self.wdbc_full = None
        self.X = None
        self.y = None
        self.X_scaled = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.label_map_string_2_int = {'M': 1, 'B': 0}

        self.features_selected = []
        self.highly_correlated_features = []
        self.cols = []
        self.class_colours = np.array(['blue', 'red', 'green', 'darkviolet', 'lime', 'darkorange', 'goldenrod',
                                       'cyan', 'silver', 'deepskyblue', 'mediumspringgreen', 'gold'])
        self.clusters_stop = 5
        self.feature_idx = {0: 0, 1: 0, 2: 0}
        self.cluster_cols = [('PerimeterMax', 'AreaSe'),
                             ('ConcavePointsMax', 'TextureMean')]

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

        with timer('\nDistributions - Full Dataset'):
            self.correlation_heatmap()
            self.update_X_drop_highly_corr()
            self.scale(self.X)
            self.distribution_multi_kde_scaled('full')
            self.distplot(self.X, 'full')
            self.kdeplot_with_target(self.wdbc_full, 'full')
            self.linearity(self.wdbc_full, 'full')
            self.cluster(self.wdbc_full, 'full', idx=self.feature_idx)

        with timer('\nOutput Wrangled Dataset'):
            self.write_csv()

        with timer('\nDistributions - Training Subset'):
            self.cols = ['ID'] + list(self.X.columns) + ['Diagnosis']
            self.X = self.wdbc_full[self.cols]
            self.set_y()
            self.train_test_split()
            self.scale(self.X_train)
            self.distribution_multi_kde_scaled('train')
            self.distplot(self.X_train, 'train')
            self.kdeplot_with_target(self.X_train, 'train')
            self.linearity(self.X_train, 'train')
            self.cluster(self.X_train, 'train', idx=self.feature_idx)

    def load_data(self):
        self.wdbc_full = pd.read_csv('data/wdbc.data', header=None)

        self.wdbc_full.columns = ['ID', 'Diagnosis', 'RadiusMean', 'TextureMean', 'PerimeterMean', 'AreaMean',
                                  'SmoothnessMean', 'CompactnessMean', 'ConcavityMean', 'ConcavePointsMean',
                                  'SymmetryMean', 'FractalDimensionMean', 'RadiusSe', 'TextureSe', 'PerimeterSe',
                                  'AreaSe', 'SmoothnessSe', 'CompactnessSe', 'ConcavitySe', 'ConcavePointsSe',
                                  'SymmetrySe', 'FractalDimensionSe', 'RadiusMax', 'TextureMax', 'PerimeterMax',
                                  'AreaMax', 'SmoothnessMax', 'CompactnessMax', 'ConcavityMax', 'ConcavePointsMax',
                                  'SymmetryMax', 'FractalDimensionMax']

        print('\n', '_' * 40, 'Shape After Data Load', '_' * 40)
        self.print_shape()

    def encode_target(self):
        self.wdbc_full['Diagnosis'] = self.wdbc_full['Diagnosis'].map(self.label_map_string_2_int)
        self.wdbc_full['Diagnosis'].astype(int)

    def column_statistics(self):
        print('\n', '_' * 40, 'Column Statistics', '_' * 40)
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
        print('\n', '_' * 40, 'Univariate Feature Selection With Chi-squared', '_' * 40)
        k = 15
        kbest = SelectKBest(score_func=chi2, k=15)
        fit = kbest.fit(self.X, self.y)
        print(fit.scores_)
        cols = kbest.get_support()
        features_selected = self.X.columns[cols]
        print(features_selected)

    def random_forest_classifier(self):
        print('\n\n', '_' * 40, 'Random Forest Classifier', '_' * 40)
        data = pd.DataFrame(columns=['Feature', 'Random Forest Importance Score'])
        k = 15
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
        plt.savefig(fname='plots/full - feature selection RFC.png', dpi=300, format='png')
        plt.show()

    def scale(self, dataset):
        df_temp = dataset.copy()
        try:
            df_temp = df_temp.drop(columns=['ID', 'Diagnosis'])
        except KeyError:
            pass
        scaler = MinMaxScaler()
        self.X_scaled = scaler.fit_transform(df_temp)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=df_temp.columns)

    def row_count_by_target(self, target):
        print('\n\n', '_' * 40, 'Row Count By {}'.format(target), '_' * 40)
        series = self.wdbc_full[target].value_counts()
        for idx, val in series.iteritems():
            print('\t{}: {} ({:6.3f}%)'.format(idx, val, ((val / self.wdbc_full.shape[0]) * 100)))

    def print_shape(self):
        print('\tRow count:\t', '{}'.format(self.wdbc_full.shape[0]))
        print('\tColumn count:\t', '{}'.format(self.wdbc_full.shape[1]))

    def distribution_multi_kde_scaled(self, dataset_name):
        for col in self.X_scaled:
            sns.kdeplot(self.X_scaled[col], shade=True)

        plt.savefig(fname='plots/' + dataset_name + ' - distplot.png', dpi=300, format='png')
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
        plt.savefig(fname='plots/full - corr heatmap top features.png', dpi=300, format='png')
        plt.show()

        # Drop highly correlated for 2nd heatmap
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        self.highly_correlated_features = [column for column in upper.columns if any(upper[column] > 0.85)]
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
        plt.savefig(fname='plots/full - corr heatmap top feat after dropping highly corr.png', dpi=300, format='png')
        plt.show()

    def update_X_drop_highly_corr(self):
        self.X = self.X.drop(self.X[self.highly_correlated_features], axis=1)
        self.features_selected = list(self.X)

    def distplot(self, dataset, dataset_name):
        for col in self.features_selected:
            if col == 'ID' or col == 'Diagnosis':
                continue
            plt.ylabel('Density')
            sns.distplot(dataset[col], hist=True, color='palevioletred', kde=True, hist_kws={'edgecolor':'black'},
                         kde_kws={'linewidth': 4})
            plt.savefig(fname='plots/' + dataset_name + ' - dist - col ' + col + '.png', dpi=300, format='png')
            plt.show()

    def kdeplot_with_target(self, dataset, dataset_name):
        for col in self.features_selected:
            sns.kdeplot(dataset.loc[dataset['Diagnosis'] == 1, col], shade=True, color='r')
            sns.kdeplot(dataset.loc[dataset['Diagnosis'] == 0, col], shade=True, color='g')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend(labels=['Malignant', 'Benign'])
            plt.savefig(fname='plots/' + dataset_name + ' - kdeplot target - ' + col + '.png', dpi=300, format='png')
            plt.show()

    def linearity(self, dataset, dataset_name):
        buckets = [0, 1]
        self.convex_hull(dataset, buckets, cola='ConcavePointsMax', colb='TextureMean', target='Diagnosis',
                         dataset_name=dataset_name)
        self.convex_hull(dataset, buckets, cola='PerimeterMax', colb='AreaSe', target='Diagnosis',
                         dataset_name=dataset_name)

    def convex_hull(self, df, buckets, cola, colb, target, dataset_name):
        cmap = plt.get_cmap('Set1')
        plt.clf()
        plt.figure(figsize=(10, 6))
        title = '{} vs {} - Label {}'.format(cola, colb, target)
        plt.title(title, fontsize=16)
        plt.xlabel(cola, fontsize=12)
        plt.ylabel(colb, fontsize=12)
        for i in range(len(buckets)):
            bucket = df[df[target] == buckets[i]]
            bucket = bucket.iloc[:, [df.columns.get_loc(cola), df.columns.get_loc(colb)]].values
            hull = ConvexHull(bucket)
            hull_color = self.class_colours[i]
            plt.scatter(bucket[:, 0], bucket[:, 1], label=buckets[i], c=self.class_colours[i], alpha=0.4)
            for j in hull.simplices:
                plt.plot(bucket[j, 0], bucket[j, 1], color=hull_color)
        plt.legend()
        plt.savefig(fname='plots/' + dataset_name + ' - convex hull - ' + cola + ' ' + colb + '.png', dpi=300,
                    format='png')
        plt.show()

    def cluster(self, dataset, dataset_name, idx):
        df_temp = dataset.copy()
        try:
            df_temp = df_temp.drop(columns=['ID', 'Diagnosis'])
        except KeyError:
            pass
        sc = StandardScaler()
        df_temp_cols = list(df_temp.columns)
        df_temp = sc.fit_transform(df_temp)
        df_temp = pd.DataFrame(df_temp, columns=df_temp_cols)
        for cola, colb in self.cluster_cols:
            for c in range(2, self.clusters_stop):
                self.set_indexes(cola, colb, df_temp)
                kmeans = KMeans(n_clusters=c, random_state=self.random_state)
                kmeans.fit(df_temp)
                y_km = kmeans.fit_predict(df_temp)
                self.scatter_clusters(df_temp, dataset_name, c, y_km, idx)

    def scatter_clusters(self, df, dataset_name, n_clusters, y_clusters, col_idx):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        df_x = df

        if isinstance(df_x, pd.DataFrame):
            df_x = df_x.values

        title = str(n_clusters) + ' Clusters - ' + df.columns[col_idx[0]] + ' vs ' + df.columns[col_idx[1]]
        plt.title(title, fontsize=12)

        xlabel = df.columns[col_idx[0]]
        ylabel = df.columns[col_idx[1]]
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # 2 clusters minimum
        for c in range(n_clusters):
            ax.scatter(df_x[y_clusters == c, col_idx[0]], df_x[y_clusters == c, col_idx[1]], alpha=0.2,
                       edgecolors='none', s=20, c=self.class_colours[c])

        plt.savefig(fname='plots/' + dataset_name + ' - ' + str(n_clusters) + ' Clusters - ' + xlabel + ' vs ' + ylabel
                          + '.png', dpi=300, format='png')
        plt.show()

    def set_indexes(self, cola, colb, dataset):
        self.feature_idx[0] = dataset.columns.get_loc(cola)
        self.feature_idx[1] = dataset.columns.get_loc(colb)

    def set_y(self):
        self.y = self.X['Diagnosis']

    def remove_target_from_X(self):
        self.X.drop('Diagnosis', axis=1, inplace=True)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30,
                                                                                random_state=self.random_state)

    def write_csv(self):
        cols = ['ID'] + list(self.X.columns) + ['Diagnosis']
        self.wdbc_full[cols].to_csv('data/wdbc_selected_cols.csv', header=True, index=False)


wdbcAnalysis = WDBCAnalysis()
