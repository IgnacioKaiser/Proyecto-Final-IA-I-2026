import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.agente import AgenteControlAcceso, obtener_particiones_impresiones

def generar_grafico_clusters():
    print("Extrayendo características para el gráfico...")
    agente = AgenteControlAcceso()
    
    X_vision = []
    particiones = obtener_particiones_impresiones()

    # 1. Usar las mismas impresiones autorizadas de entrenamiento que el agente.
    if particiones:
        rutas = [ruta for ruta, _ in particiones["entrenamiento"]]
    else:
        directorio_autorizadas = "data/huellas/autorizadas"
        if not os.path.exists(directorio_autorizadas):
            raise FileNotFoundError("No se encontró la carpeta de huellas autorizadas.")
        rutas = [
            os.path.join(directorio_autorizadas, carpeta_id, archivo)
            for carpeta_id in os.listdir(directorio_autorizadas)
            for archivo in os.listdir(os.path.join(directorio_autorizadas, carpeta_id))
            if os.path.isfile(os.path.join(directorio_autorizadas, carpeta_id, archivo))
            and archivo.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    for ruta_imagen in rutas:
        esqueleto = agente.preprocesar_huella(str(ruta_imagen))
        caracteristicas = agente.extraer_caracteristicas_huella(esqueleto)
        X_vision.append(caracteristicas)

    X_vision = np.array(X_vision)
    
    if len(X_vision) < 3:
        print("Faltan datos para graficar.")
        return

    # 2. Normalización Z-Score de los descriptores HOG.
    scaler = StandardScaler()
    X_vision_escalado = scaler.fit_transform(X_vision)

    # 3. Entrenar K-Means con los datos normalizados.
    agente.modelo_vision.fit(X_vision_escalado)
    centroides_7d = agente.modelo_vision.cluster_centers_
    predicciones = agente.modelo_vision.labels_

    # 4. Reducción de dimensionalidad con PCA para visualizar los clusters.
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_vision_escalado)
    centroides_2d = pca.transform(centroides_7d)

    # 5. Configurar el gráfico con Matplotlib
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

    # Formateo profesional para el informe académico
    plt.title('Agrupamiento K-Means de Huellas Dactilares (Reducción PCA a 2D)', fontsize=14, fontweight='bold')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% de la varianza)', fontsize=12)
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% de la varianza)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 6. Guardar la imagen en alta resolución
    nombre_archivo = "grafico_kmeans_informe.png"
    plt.savefig(nombre_archivo, dpi=300)
    print(f"✅ Gráfico generado y guardado exitosamente como '{nombre_archivo}'")
    
    # Mostrarlo en pantalla
    plt.show()

if __name__ == "__main__":
    generar_grafico_clusters()