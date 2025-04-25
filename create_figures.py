import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, Isomap
from ucimlrepo import fetch_ucirepo
import seaborn as sns
import glob


def dim_reduce(dimension, algorithm, dataset):
    if algorithm == 'PCA':
        pca = PCA(n_components=dimension)
        reduced_data = pca.fit_transform(dataset)
    elif algorithm == 'TSNE':
        tsne = TSNE(n_components=dimension)
        reduced_data = tsne.fit_transform(dataset)
    elif algorithm == 'Isomap':
        isomap = Isomap(n_components=dimension)
        reduced_data = isomap.fit_transform(dataset)
    elif algorithm == 'ICA':
        ica = FastICA(n_components=dimension)
        reduced_data = ica.fit_transform(dataset)
    else:
        raise ValueError("Unknown algorithm")

    return pd.DataFrame(reduced_data, index=dataset.index)


def cluster(n_cluster_or_eps, algorithm, dataset):
    if algorithm == 'KMeans':
        model = KMeans(n_clusters=n_cluster_or_eps, random_state=42)
        labels = model.fit_predict(dataset)
    elif algorithm == 'GaussianMixture':
        model = GaussianMixture(n_components=n_cluster_or_eps, random_state=42)
        labels = model.fit_predict(dataset)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=n_cluster_or_eps, min_samples=5)
        labels = model.fit_predict(dataset)
    elif algorithm == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_cluster_or_eps)
        labels = model.fit_predict(dataset)
    else:
        raise ValueError("Unknown algorithm")

    return pd.DataFrame(labels, index=dataset.index, columns=['cluster'])


# Load Data, Data Dictionary, and Results
response = fetch_ucirepo(id=891)
data = response['data']['features']
labels = response['data']['targets']
data_dictionary = pd.read_csv('data_dictionary.csv', index_col=[0, 1])
data_dictionary.index = data_dictionary.index.droplevel(0)
results = pd.read_csv("results.csv")


# Load clustering and dimensionality reduction results
Agg_8 = pd.read_csv('reduced_subset_1_ICA_3_Transformed_Agglomerative_8_cluster.csv', index_col=0)
Agg_4 = pd.read_csv('reduced_subset_1_ICA_3_Transformed_Agglomerative_4_cluster.csv', index_col=0)
KMeans_8 = pd.read_csv('reduced_subset_1_ICA_3_Transformed_KMeans_8_cluster.csv', index_col=0)
KMeans_4 = pd.read_csv('reduced_subset_1_ICA_3_Transformed_KMeans_4_cluster.csv', index_col=0)
ICA_3 = pd.read_csv('reduced_subset_1_ICA_3_Transformed.csv', index_col=0)
reduced_ICA_3 = dim_reduce(2, 'TSNE', ICA_3)
agg_8_clusterings = glob.glob('reduced_subset_*_ICA_3_Transformed_Agglomerative_8_cluster.csv')
agg_8_clusterings_labels = [pd.read_csv(file, index_col=0) for file in agg_8_clusterings]


# Create Figures
plt.style.use('ggplot')

# Figure 1
a = results.loc[results['transformed'], 'silhouette_score'].dropna()
b = results.loc[~results['transformed'], 'silhouette_score'].dropna()
c = results.loc[results['transformed'], 'mutual_info_score'].dropna()
d = results.loc[~results['transformed'], 'mutual_info_score'].dropna()
a1 = results.loc[results['transformed'] & (results['clustering_alg'] == 'DBSCAN'), 'silhouette_score']
b1 = results.loc[~results['transformed'] & (results['clustering_alg'] == 'DBSCAN'), 'silhouette_score']
a1 = a1.dropna()
b1 = b1.dropna()
a2 = results.loc[results['transformed'] & (results['clustering_alg'] == 'KMeans'), 'silhouette_score']
b2 = results.loc[~results['transformed'] & (results['clustering_alg'] == 'KMeans'), 'silhouette_score']
a2 = a2.dropna()
b2 = b2.dropna()
a3 = results.loc[results['transformed'] & (results['clustering_alg'] == 'GaussianMixture'), 'silhouette_score']
b3 = results.loc[~results['transformed'] & (results['clustering_alg'] == 'GaussianMixture'), 'silhouette_score']
a3 = a3.dropna()
b3 = b3.dropna()
a4 = results.loc[results['transformed'] & (results['clustering_alg'] == 'Agglomerative'), 'silhouette_score']
b4 = results.loc[~results['transformed'] & (results['clustering_alg'] == 'Agglomerative'), 'silhouette_score']
a4 = a4.dropna()
b4 = b4.dropna()
c1 = results.loc[results['transformed'] & (results['clustering_alg'] == 'DBSCAN'), 'mutual_info_score']
d1 = results.loc[~results['transformed'] & (results['clustering_alg'] == 'DBSCAN'), 'mutual_info_score']
c1 = c1.dropna()
d1 = d1.dropna()
c2 = results.loc[results['transformed'] & (results['clustering_alg'] == 'KMeans'), 'mutual_info_score']
d2 = results.loc[~results['transformed'] & (results['clustering_alg'] == 'KMeans'), 'mutual_info_score']
c2 = c2.dropna()
d2 = d2.dropna()
c3 = results.loc[results['transformed'] & (results['clustering_alg'] == 'GaussianMixture'), 'mutual_info_score']
d3 = results.loc[~results['transformed'] & (results['clustering_alg'] == 'GaussianMixture'), 'mutual_info_score']
c3 = c3.dropna()
d3 = d3.dropna()
c4 = results.loc[results['transformed'] & (results['clustering_alg'] == 'Agglomerative'), 'mutual_info_score']
d4 = results.loc[~results['transformed'] & (results['clustering_alg'] == 'Agglomerative'), 'mutual_info_score']
c4 = c4.dropna()
d4 = d4.dropna()
fig, axes = plt.subplots(2, 2, figsize=(7, 6))
axes[0, 1].boxplot([a1, b1, a2, b2, a3, b3, a4, b4], tick_labels=['DBSCAN\nTransformed', 'DBSCAN\nUntransformed', 'K-Means\nTransformed', 'K-Means\nUntransformed', 'GaussianMixture\nTransformed', 'GaussianMixture\nUntransformed', 'Hierarchical Clustering\nTransformed', 'Hierarchical Clustering\nUntransformed'])
axes[1, 1].boxplot([c1, d1, c2, d2, c3, d3, c4, d4], tick_labels=['DBSCAN\nTransformed', 'DBSCAN\nUntransformed', 'K-Mean\nTransformed', 'K-Means\nUntransformed', 'GaussianMixture\nTransformed', 'GaussianMixture\nUntransformed', 'Hierarchical Clustering\nTransformed', 'Hierarchical Clustering\nUntransformed'])
axes[0, 0].boxplot([a, b], tick_labels=['Transformed', 'Untransformed'])
axes[0, 0].set_ylabel('Silhouette Score')
axes[1, 0].set_ylabel('Normalized MI')
axes[1, 0].boxplot([c, d], tick_labels=['Transformed', 'Untransformed'])
for ax in axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=90, ha='right')
axes[0, 0].text(-0.05, 1.05, '(A)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[0, 0].transAxes)
axes[0, 1].text(-0.05, 1.05, '(B)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[0, 1].transAxes)
axes[1, 0].text(-0.05, 1.05, '(C)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[1, 0].transAxes)
axes[1, 1].text(-0.05, 1.05, '(D)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[1, 1].transAxes)
fig.tight_layout()
fig.savefig('fig1.svg')
print('Figure 1 saved as fig1.svg')


# Figure 2
table = results.loc[results['transformed']].groupby(['dim_reduce_alg', 'dim_reduce_param'])['silhouette_score'].mean().unstack().drop(index='TSNE')
table2 = results.loc[results['transformed']].groupby(['dim_reduce_alg', 'dim_reduce_param'])['mutual_info_score'].mean().unstack().drop(index='TSNE')
mask1 = results['transformed'] & (results['dim_reduce_alg'] == 'PCA') & (results['dim_reduce_param'] == 3)
mask2 = results['transformed'] & (results['dim_reduce_alg'] == 'ICA') & (results['dim_reduce_param'] == 3)
mask3 = results['transformed'] & (results['dim_reduce_alg'] == 'PCA') & (results['dim_reduce_param'] == 5)
mask4 = results['transformed'] & (results['dim_reduce_alg'] == 'ICA') & (results['dim_reduce_param'] == 5)
a = results.loc[mask1, 'silhouette_score'].dropna()
b = results.loc[mask2, 'silhouette_score'].dropna()
c = results.loc[mask3, 'silhouette_score'].dropna()
d = results.loc[mask4, 'silhouette_score'].dropna()
a1 = results.loc[mask1, 'mutual_info_score'].dropna()
b1 = results.loc[mask2, 'mutual_info_score'].dropna()
c1 = results.loc[mask3, 'mutual_info_score'].dropna()
d1 = results.loc[mask4, 'mutual_info_score'].dropna()
fig, axes = plt.subplots(2, 2, figsize=(7, 4))
sns.heatmap(table, annot=True, fmt='.2f', cmap='coolwarm', cbar=False, ax=axes[0, 0])
axes[0, 0].yaxis.set_label_text('')
sns.heatmap(table2, annot=True, fmt='.2%', cmap='coolwarm', cbar=False, ax=axes[0, 1])
axes[0, 1].yaxis.set_label_text('')
axes[1, 0].boxplot([a, b, c, d], tick_labels=['PCA-3', 'ICA-3', 'PCA-5', 'ICA-5'])
axes[1, 0].set_ylabel('Silhouette Score')
axes[1, 1].boxplot([a1, b1, c1, d1], tick_labels=['PCA-3', 'ICA-3', 'PCA-5', 'ICA-5'])
axes[1, 1].set_ylabel('Normalized MI')
axes[0, 0].text(-0.1, 1.05, '(A)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[0, 0].transAxes)
axes[0, 1].text(-0.1, 1.05, '(B)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[0, 1].transAxes)
axes[1, 0].text(-0.1, 1.05, '(C)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[1, 0].transAxes)
axes[1, 1].text(-0.1, 1.05, '(D)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[1, 1].transAxes)
fig.tight_layout()
fig.savefig('fig2.svg')
print('Figure 2 saved as fig2.svg')


# Figure 3
mask1 = (results['dim_reduce_alg'] == 'PCA') & (results['dim_reduce_param'] == 3) & (results['transformed'] == True)
mask2 = (results['dim_reduce_alg'] == 'ICA') & (results['dim_reduce_param'] == 3) & (results['transformed'] == True)
mask = mask1 | mask2
fig, axes = plt.subplots(2, 2, figsize=(7, 4), sharey='row')
a1 = results.loc[mask & (results['clustering_alg'] == 'KMeans') & (results['clustering_param'] == 8), 'mutual_info_score'].dropna()
b1 = results.loc[mask & (results['clustering_alg'] == 'Agglomerative') & (results['clustering_param'] == 7), 'mutual_info_score'].dropna()
c1 = results.loc[mask & (results['clustering_alg'] == 'Agglomerative') & (results['clustering_param'] == 8), 'mutual_info_score'].dropna()
d1 = results.loc[mask & (results['clustering_alg'] == 'Agglomerative') & (results['clustering_param'] == 6), 'mutual_info_score'].dropna()
e1 = results.loc[mask & (results['clustering_alg'] == 'KMeans') & (results['clustering_param'] == 7), 'mutual_info_score'].dropna()
f1 = results.loc[mask & (results['clustering_alg'] == 'KMeans') & (results['clustering_param'] == 6), 'mutual_info_score'].dropna()
g1 = results.loc[mask & (results['clustering_alg'] == 'KMeans') & (results['clustering_param'] == 5), 'mutual_info_score'].dropna()
axes[1, 0].boxplot([a1, b1, c1, d1, e1, f1, g1], tick_labels=['K-Means-8', 'Hirerarchical-7', 'Hirerarchical-8', 'Hirerarchical-6', 'K-Means-7', 'K-Means-6', 'K-Means-5'], orientation='horizontal')
axes[1, 0].xaxis.set_tick_params(rotation=90)
axes[1, 0].set_xlabel('Normalized MI')
axes[1, 0].text(-0.1, 1.1, '(C)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[1, 0].transAxes)
a = results.loc[mask & (results['clustering_alg'] == 'DBSCAN') & (results['clustering_param'] == 10), 'silhouette_score'].dropna()
b = results.loc[mask & (results['clustering_alg'] == 'Agglomerative') & (results['clustering_param'] == 4), 'silhouette_score'].dropna()
c = results.loc[mask & (results['clustering_alg'] == 'KMeans') & (results['clustering_param'] == 4), 'silhouette_score'].dropna()
d = results.loc[mask & (results['clustering_alg'] == 'GaussianMixture') & (results['clustering_param'] == 4), 'silhouette_score'].dropna()
e = results.loc[mask & (results['clustering_alg'] == 'Agglomerative') & (results['clustering_param'] == 5), 'silhouette_score'].dropna()
axes[0, 1].boxplot([a, b, c, d, e], tick_labels=['DBSCAN-10', 'Hirerarchical-4', 'K-Means-4', 'GMM-4', 'Hirerarchical-3'], orientation='horizontal')
axes[0, 1].xaxis.set_tick_params(rotation=90)
axes[0, 1].set_xlabel('Silhouette Score')
axes[0, 1].text(-0.1, 1.1, '(B)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[0, 1].transAxes)
a1 = results.loc[mask & (results['clustering_alg'] == 'DBSCAN') & (results['clustering_param'] == 10), 'mutual_info_score'].dropna()
b1 = results.loc[mask & (results['clustering_alg'] == 'Agglomerative') & (results['clustering_param'] == 4), 'mutual_info_score'].dropna()
c1 = results.loc[mask & (results['clustering_alg'] == 'KMeans') & (results['clustering_param'] == 4), 'mutual_info_score'].dropna()
d1 = results.loc[mask & (results['clustering_alg'] == 'GaussianMixture') & (results['clustering_param'] == 4), 'mutual_info_score'].dropna()
e1 = results.loc[mask & (results['clustering_alg'] == 'Agglomerative') & (results['clustering_param'] == 5), 'mutual_info_score'].dropna()
axes[0, 0].boxplot([a1, b1, c1, d1, e1], tick_labels=['DBSCAN-10', 'Hirerarchical-4', 'K-Means-4', 'GMM-4', 'Hirerarchical-3'], orientation='horizontal')
axes[0, 0].set_xlabel('Normalized MI')
axes[0, 0].xaxis.set_tick_params(rotation=90)
axes[0, 0].text(-0.1, 1.1, '(A)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[0, 0].transAxes)
axes[1, 1].set_visible(False)
fig.tight_layout()
fig.savefig('fig3.svg')
print('Figure 3 saved as fig3.svg')


# Figure 4
fig, axes = plt.subplots(2, 2, figsize=(7, 4))
axes[0, 0].scatter(reduced_ICA_3[0], reduced_ICA_3[1], s=50, alpha=0.1, c=Agg_8['cluster'].values, cmap='viridis', edgecolors='none')
axes[0, 0].scatter(reduced_ICA_3[0], reduced_ICA_3[1], c=labels.loc[reduced_ICA_3.index].values, sizes=0.1*np.ones(len(reduced_ICA_3)), cmap='jet')
axes[0, 0].text(-0.1, 1.1, '(A)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[0, 0].transAxes)
axes[1, 0].scatter(reduced_ICA_3[0], reduced_ICA_3[1], s=50, alpha=0.1, c=Agg_4['cluster'].values, cmap='viridis', edgecolors='none')
axes[1, 0].scatter(reduced_ICA_3[0], reduced_ICA_3[1], c=labels.loc[reduced_ICA_3.index].values, sizes=0.1*np.ones(len(reduced_ICA_3)), cmap='jet')
axes[1, 0].text(-0.1, 1.1, '(C)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[1, 0].transAxes)
axes[0, 1].scatter(reduced_ICA_3[0], reduced_ICA_3[1], s=50, alpha=0.1, c=KMeans_8['cluster'].values, cmap='viridis', edgecolors='none')
axes[0, 1].scatter(reduced_ICA_3[0], reduced_ICA_3[1], c=labels.loc[reduced_ICA_3.index].values, sizes=0.1*np.ones(len(reduced_ICA_3)), cmap='jet')
axes[0, 1].text(-0.1, 1.1, '(B)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[0, 1].transAxes)
axes[1, 1].scatter(reduced_ICA_3[0], reduced_ICA_3[1], s=50, alpha=0.1, c=KMeans_4['cluster'].values, cmap='viridis', edgecolors='none')
axes[1, 1].scatter(reduced_ICA_3[0], reduced_ICA_3[1], c=labels.loc[reduced_ICA_3.index].values, sizes=0.1*np.ones(len(reduced_ICA_3)), cmap='jet')
axes[1, 1].text(-0.1, 1.1, '(D)', ha='center', va='center', fontsize=18, fontweight='bold', transform=axes[1, 1].transAxes)
fig.tight_layout()
fig.savefig('fig4.pdf')
print('Figure 4 saved as fig4.svg')


# Figure 5
total_Agg_8 = dim_reduce(3, 'ICA', data)
total_Kmeans_8 = cluster(8, 'KMeans', total_Agg_8)
cols = data_dictionary[data_dictionary['type'].isin(['binary', 'ordinal'])].index
fig, ax = plt.subplots(figsize=(7, 4))
things = {}
for col in cols:
    sum = []
    for cluster in agg_8_clusterings_labels:
        nmi = normalized_mutual_info_score(data.loc[cluster.index, col].values.reshape(-1), cluster['cluster'].values.reshape(-1))
        sum.append(nmi)
    things[col] = sum
ax.boxplot(things.values(), tick_labels=things.keys())
ax.xaxis.set_tick_params(rotation=90)
ax.set_ylabel('Normalized MI')
fig.tight_layout()
fig.savefig('fig5.svg')
print('Figure 5 saved as fig5.svg')


# Figure 6
my_table = total_Kmeans_8.join(labels, how='left').join(data[['BMI', 'Smoker']], how='left').groupby('cluster')[['BMI', 'Diabetes_binary', 'Smoker']].agg({'BMI': 'mean', 'Diabetes_binary': 'mean', 'Smoker': 'count'})
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(my_table['BMI'], my_table['Diabetes_binary'], s=np.sqrt(my_table['Smoker']) * 0.5, alpha=0.5, c=my_table.index, cmap='viridis', edgecolors='none')
n = my_table.index + 1
for i, txt in enumerate(n):
    ax.annotate(txt, (my_table['BMI'].iloc[i] - 0.8, my_table['Diabetes_binary'].iloc[i] + 0.005), fontsize=8, ha='center', va='center')
ax.set_xlabel('BMI')
ax.set_ylabel('Probabilty of Diabetes')
fig.tight_layout()
fig.savefig('fig6.svg')
print('Figure 6 saved as fig6.svg')
