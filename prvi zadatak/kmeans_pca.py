from data_preprocessing import load_and_preprocess
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def main():
    X = load_and_preprocess()

    # PCA
    min_clusters = 2
    max_clusters = 8

    min_pc = 5
    max_pc = 9
    var_ratio = {}
    """for y in range(min_pc, max_pc):
        print("PCA with # of components: ", y)
        pca = PCA(n_components=y)
        fitted = pca.fit(X)
        data_p = fitted.fit_transform(X)
        var_ratio[y] = sum(fitted.explained_variance_ratio_)
        for x in range(min_clusters, max_clusters):
            alg = KMeans(n_clusters=x)
            label = alg.fit_predict(data_p)
            print('Silhouette-Score for', x, 'Clusters: ', silhouette_score(data_p, label), '       Inertia: ',
                  alg.inertia_)
        print()
    print(var_ratio)"""

    #we chose 5 pcs as they explain 80% of the variance
    pca = PCA(n_components=5)
    reduced_dataframe = pca.fit_transform(X)
    silhouette = []
    inertia = []
    for x in range(min_clusters, max_clusters):
        alg = KMeans(n_clusters=x)
        label = alg.fit_predict(reduced_dataframe)
        inertia.append(alg.inertia_)
        silhouette.append(silhouette_score(reduced_dataframe, label))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
    x = range(min_clusters, max_clusters)

    axes[0].set_title("The inertia plot for the various clusters.")
    axes[0].set_xlabel("The inertia values")
    axes[0].set_ylabel("Cluster label")
    axes[0].plot(x, inertia)

    axes[1].set_title("The silhouette plot for the various clusters.")
    axes[1].set_xlabel("The silhouette coefficient values")
    axes[1].set_ylabel("Cluster label")
    axes[1].plot(x, silhouette)

    plt.show()

    #according to plots, chose 4,5 or 6 clusters
if __name__ == '__main__':
    main()