from ast import Num
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

def optimize_kmeans(data, _range):
    """
    Optimiza el número de clusters en KMeans a través de la inercia.
    Utiliza el método del codo para determinar el número óptimo de clusters.
    Genera un gráfico para visualizar los valores de inercia.

    Parametros
    ----
    - data: list, Datos de entrada para la optimización.
    - _range: list, Rango de números de clusters a evaluar (iterable).
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
    Genera el agrupamiento KMeans para una lista de números de clusters.
    
    Parámetros
    ----
    - data: list, Datos de entrada para la agrupación.
    - clusters: list, Lista de números de clusters a evaluar (iterable).
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
    Grafica los clusters y sus centroides.

    Parámetros
    ----
    - data: list, Array 2D de puntos de datos a agrupar.
    - k_labels: list, Array 2D de etiquetas de cluster para cada punto de datos.
        Cada columna corresponde a un modelo KMeans diferente.
    - k_menas: list, Array 1D con el número de clusters para cada modelo KMeans.
        Cada elemento corresponde al número de clusters para cada modelo KMeans.
    """

    fig, axs = plt.subplots(figsize=(10, 5), nrows=1, ncols=len(k_labels[0]))
    for i, ax in enumerate(fig.axes):
        ax.scatter(data[:, 0], data[:, 2], c=k_labels[:, i])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_title(f'KMeans con {k_menas[i]} clusters')
        ax.grid(True)

    fig.canvas.manager.set_window_title('KMeans Clusters')
    plt.show()

if __name__ == "__main__":
    # Importar datos
    data = np.genfromtxt(
        'assets/muestra4s.csv',
        delimiter=',',
        skip_header=1, # Salta los encabezados
        dtype=np.float32, # Cambia el tipo de datos a float32
    )
    print(f'Data loaded: {data.shape}')

    # Normalizar datos
    normalized_data = Normalizer().transform(data[:, 1:])
    print(f'Data normalized: {normalized_data.shape}')
    
    # Lista de clusters
    clusters_values = [3, 5, 15]
    # Entrenar KMeans
    clusters = get_k_means(normalized_data, clusters_values)
    # Graficar KMeans
    plot_clusters(normalized_data, clusters['labels'], clusters['menas'])

