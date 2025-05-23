import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from minisom import MiniSom

def kohonen_net(
    data,
    x_dim=10,
    y_dim=10,
    sigma=1.0,
    learning_rate=0.5,
    num_iter=1000
):
    """
    Entrena una red de Kohonen (SOM) con los datos proporcionados.

    Parámetros:
    ----------
    data: np.ndarray, matriz de datos (n_samples x n_features)
    x_dim: int, dimensión horizontal del mapa SOM
    y_dim: int, dimensión vertical del mapa SOM
    sigma: float, radio de influencia inicial
    learning_rate: float, tasa de aprendizaje inicial
    num_iter: int, número de iteraciones de entrenamiento

    Retorna:
    - som: objeto MiniSom entrenado
    """

    # Crear el SOM
    som = MiniSom(x_dim, y_dim, data.shape[1], sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(data)

    # Entrenar el SOM
    som.train_random(data, num_iter)

    return som

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

    # Entrenar la red de Kohonen
    som = kohonen_net(normalized_data)
    print(f'SOM trained with dimensions: {som.get_weights().shape}')

    # Visualizar los pesos
    plt.figure(figsize=(7, 7), num='SOM')
    # Visualizar el mapa de distancias (U-Matrix)
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar(label='Peso')
    plt.title('Mapa de pesos')
    plt.show()