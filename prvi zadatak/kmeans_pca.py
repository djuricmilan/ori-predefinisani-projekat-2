from data_preprocessing import load_and_preprocess
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import math

def calculate_kn_distance(X,k):

    kn_distance = []
    for i in range(len(X)):
        eucl_dist = []
        for j in range(len(X)):
            eucl_dist.append(
                math.sqrt(
                    ((X[i,0] - X[j,0]) ** 2) +
                    ((X[i,1] - X[j,1]) ** 2)))

        eucl_dist.sort()
        kn_distance.append(eucl_dist[k])

    return kn_distance

suma = 0
cnt = 0

def avg_cluster(row):
    global suma, cnt
    suma += row['PURCHASES_FREQUENCY'] if row['CLUSTER_5'] == 2 else 0
    cnt += 1 if row['CLUSTER_5'] == 2 else 0

def main():
    _, X = load_and_preprocess()

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

    #test number of clusters
    """silhouette = []
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

    plt.show()"""

    # according to plots, chose 4,5 or 6 clusters
    #we chose 5 clusters

    alg = KMeans(n_clusters=5)
    label = alg.fit_predict(reduced_dataframe)

    # *** ALTERNATIVE ALGORITHMS ***
    #eps_dist = calculate_kn_distance(reduced_dataframe, 10)
    #plt.hist(eps_dist, bins=30)
    #plt.ylabel('n')
    #plt.xlabel('Epsilon distance')

    #dbscan = DBSCAN(eps=1, min_samples=10)
    #label = dbscan.fit_predict(reduced_dataframe)

    #ms = MeanShift()
    #label = ms.fit_predict(reduced_dataframe)

    #aggloerative = AgglomerativeClustering(n_clusters=5)
    #label = aggloerative.fit_predict(reduced_dataframe)

    #birch = Birch(threshold=0.3)
    #label = birch.fit_predict(reduced_dataframe)


    fig, ax = plt.subplots()
    ax.set_title("5 clusters according to 5 principle components.")
    ax.set_xlabel("PC 0")
    ax.set_ylabel("PC 1")
    plt.scatter(reduced_dataframe[:, 0], reduced_dataframe[:, 1], c=label.astype(float)) #visualize clustering according to first two prinicple components
    plt.show()

    chosen_columns = ["BALANCE_TO_CREDIT_LIMIT_RATIO", "PAYMENT_TO_MIN_PAYMENT_RATIO", "MONTHLY_AVERAGE_PURCHASES",
                      "MONTHLY_AVERAGE_CASH_ADVANCE"]
    X_chosen = pd.DataFrame(X[chosen_columns])
    # create a 'cluster' column

    X_chosen['CLUSTER_5'] = label
    chosen_columns.append('CLUSTER_5')
    # make a Seaborn pairplot
    # sb.pairplot(X_chosen, hue='CLUSTER_5', vars=chosen_columns, diag_kind='kde', size=3)

    # ALL FEATURES ALL CLUSTERS
    dataframe = pd.read_csv("credit_card_data.csv")
    dataframe["BALANCE_TO_CREDIT_LIMIT_RATIO"] = dataframe["BALANCE"] / dataframe["CREDIT_LIMIT"]

    # 2. PAYMENTS TO MINIMUM PAYMENTS RATIO - SHOULD BE HIGH IF CREDIT CARD USER IS RESPONSIBLE
    dataframe["PAYMENT_TO_MIN_PAYMENT_RATIO"] = dataframe["PAYMENTS"] / dataframe["MINIMUM_PAYMENTS"]

    # 3. MONTHLY AVERAGE PURCHASES AND CASH ADVANCE - IMPORTANT SINCE TENURE VARIES SIGNIFICANTLY AMONG CREDIT CARD HOLDERS
    dataframe["MONTHLY_AVERAGE_PURCHASES"] = dataframe["PURCHASES"] / dataframe["TENURE"]
    dataframe["MONTHLY_AVERAGE_CASH_ADVANCE"] = dataframe["CASH_ADVANCE"] / dataframe["TENURE"]
    dataframe['CLUSTER_5'] = label
    dataframe.apply(avg_cluster, axis=1)
    global suma, cnt
    print(suma/cnt)
    for c in dataframe:
        grid= sb.FacetGrid(dataframe, col='CLUSTER_5')
        grid.map(plt.hist, c)
    plt.show()



if __name__ == '__main__':
    main()