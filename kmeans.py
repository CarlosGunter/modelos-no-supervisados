import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

def optimize_kmeans(data, _range):
    """
    Optimizes KMeans clustering by calculating inertia for a range of cluster numbers.
    
    Generates a plot to visualize the inertia values.

    Parameters
    ----
    data: list
        Input data for clustering.
    clusters: list
        Range of cluster numbers to evaluate (iterable).
    """

    innertias = {   
        'menas': np.zeros(len(_range)),
        'inertias': np.zeros(len(_range)),
    }
    for i, k in enumerate(_range):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        innertias['menas'][i] = kmeans.n_clusters
        innertias['inertias'][i] = kmeans.inertia_
        
    fig = plt.subplots(figsize=(10, 5))
    plt.plot(innertias['menas'], innertias['inertias'], 'o-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()

# KMeans clustering
def get_k_means(data, clusters):
    """
    Generates KMeans clustering for a list of cluster numbers.
    
    Parameters
    ----
    data: list
        Input data for clustering.
    clusters: list
        List of cluster numbers to evaluate (iterable).
    """

    results = {
        'menas': np.zeros(len(clusters)),
        'labels': np.zeros((data.shape[0], len(clusters))),
    }
    for i, k in enumerate(clusters):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        results['menas'][i] = kmeans.n_clusters
        results['labels'][:, i] = kmeans.labels_

    return results

def plot_clusters(data, k_labels, k_menas):
    """
    Plots the clusters and their centroids.

    Parameters
    ----
    data: list
        2D array of data points to be clustered.
    k_labels: list
        2D array of cluster labels for each data point.
        Each column corresponds to a different KMeans model.
    k_menas: list
        1D array of cluster numbers for each KMeans model.
        Each element corresponds to the number of clusters for each KMeans model.
    """

    fig, axs = plt.subplots(figsize=(10, 5), nrows=1, ncols=len(k_labels[0]))
    for i, ax in enumerate(fig.axes):
        ax.scatter(data[:, 0], data[:, 2], c=k_labels[:, i])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title(f'KMeans with {k_menas[i]} clusters')
        ax.grid(True)

    plt.show()

if __name__ == "__main__":
    # Load the data
    data = np.genfromtxt(
        'assets/muestra4s.csv',
        delimiter=',',
        skip_header=1, # Salta los encabezados
        dtype=np.float32, # Cambia el tipo de datos a float32
    )
    print(f'Data loaded: {data.shape}')

    # Normalize data
    normalized_data = Normalizer().transform(data[:, 1:])
    print(f'Data normalized: {normalized_data.shape}')
    
    # Cluster values
    clusters_values = [3, 5, 15]
    # Get KMeans clustering
    clusters = get_k_means(normalized_data, clusters_values)
    # KMeans plots
    plot_clusters(normalized_data, clusters['labels'], clusters['menas'])

