import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.agente import AgenteControlAcceso

def generar_grafico_clusters():
    print("Extrayendo características para el gráfico...")
    agente = AgenteControlAcceso()
    
    directorio_autorizadas = "data/huellas/autorizadas"
    X_vision = []
    etiquetas_reales = []
    nombres_clases = []

    # 1. Recolectar datos usando los métodos del agente
    if not os.path.exists(directorio_autorizadas):
        print("Error: No se encontró la carpeta de huellas autorizadas.")
        return

    for idx, carpeta_id in enumerate(os.listdir(directorio_autorizadas)):
        ruta_id = os.path.join(directorio_autorizadas, carpeta_id)
        if os.path.isdir(ruta_id):
            nombres_clases.append(carpeta_id)
            for archivo in os.listdir(ruta_id):
                if archivo.endswith(('.png', '.jpg')):
                    ruta_imagen = os.path.join(ruta_id, archivo)
                    try:
                        esqueleto = agente.preprocesar_huella(ruta_imagen)
                        caracteristicas = agente.extraer_caracteristicas_huella(esqueleto)
                        X_vision.append(caracteristicas)
                        etiquetas_reales.append(idx)
                    except Exception as e:
                        pass

    X_vision = np.array(X_vision)
    
    if len(X_vision) < 3:
        print("Faltan datos para graficar.")
        return

    # 2. Entrenar K-Means con los datos originales (7D)
    agente.modelo_vision.fit(X_vision)
    centroides_7d = agente.modelo_vision.cluster_centers_
    predicciones = agente.modelo_vision.labels_

    # 3. Reducción de dimensionalidad con PCA (de 7D a 2D)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_vision)
    centroides_2d = pca.transform(centroides_7d)

    # 4. Configurar el gráfico con Matplotlib
    plt.figure(figsize=(10, 7))
    
    # Colores para los 3 clusters
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Graficar las muestras de cada cluster
    for k in range(3):
        puntos_cluster = X_2d[predicciones == k]
        plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], 
                    c=colores[k], label=f'Cluster {k+1}', 
                    alpha=0.6, edgecolors='w', s=80)

    # Graficar los centroides
    plt.scatter(centroides_2d[:, 0], centroides_2d[:, 1], 
                c='red', marker='X', s=200, label='Centroides', 
                edgecolors='black', linewidths=2)

    # Formateo profesional para el informe
    plt.title('Agrupamiento K-Means de Huellas Dactilares (Reducción PCA a 2D)', fontsize=14, fontweight='bold')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% de la varianza)', fontsize=12)
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% de la varianza)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 5. Guardar la imagen
    nombre_archivo = "grafico_kmeans_informe.png"
    plt.savefig(nombre_archivo, dpi=300) # dpi=300 asegura alta resolución para el PDF
    print(f"✅ Gráfico generado y guardado exitosamente como '{nombre_archivo}'")
    
    # Mostrarlo en pantalla
    plt.show()

if __name__ == "__main__":
    generar_grafico_clusters()