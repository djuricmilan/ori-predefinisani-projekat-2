from sklearn.cluster import KMeans
from data_preprocessing import load_and_preprocess
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

def main():
    min_clusters = 2
    max_clusters = 15
    X = load_and_preprocess()
    inertia = []

    #STORE INERTIA OF EACH CLUSTER IN ORDER TO VISUALIZE IT LATER (THE ELBOW METHOD)
    for i in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.plot(range(min_clusters, max_clusters + 1), inertia, marker='x')
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Inertia plot (The elbow method)")
    plt.show()

    #CALCULATE SILHOUETTE SCORES FOR EACH CLUSTER
    max_clusters=8
    for i in range(min_clusters, max_clusters + 1):
        #obtain axes
        fig, (axes) = plt.subplots(1, 1)

        # The silhouette coefficient can range from -1, 1
        axes.set_xlim([-1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        axes.set_ylim([0, len(X) + (i + 1) * 10])

        #perform k menas for i clusters
        clusterer = KMeans(n_clusters=i)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For i =", i,
              "clusters, the average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for j in range(i):
            # Aggregate the silhouette scores for samples belonging to
            # cluster j, and sort them
            jth_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == j]

            jth_cluster_silhouette_values.sort()

            size_cluster_j = jth_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j

            color = cm.nipy_spectral(float(j) / i)
            axes.fill_betweenx(np.arange(y_lower, y_upper),
                              0, jth_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            axes.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        axes.set_title("The silhouette plot for the various clusters.")
        axes.set_xlabel("The silhouette coefficient values")
        axes.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        axes.axvline(x=silhouette_avg, color="red", linestyle="--")

        axes.set_yticks([])  # Clear the yaxis labels / ticks
        axes.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()


if __name__ == '__main__':
    main()