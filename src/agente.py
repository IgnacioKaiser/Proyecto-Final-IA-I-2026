import numpy as np
import cv2
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from collections import Counter

from src.procesamiento_voz import cargar_dataset_voz, escuchar_audio_en_vivo
from src.procesamiento_imagen import procesar_pipeline_completo

class AgenteControlAcceso:
    def __init__(self, umbral_aceptacion=1.5): # <-- Umbral más estricto por la normalización
        self.modelo_vision = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.centroides = None
        self.umbral_aceptacion = umbral_aceptacion
        self.scaler_vision = StandardScaler()
        self.mapa_clusters = {} # <-- NUEVO: Diccionario para vincular K-Means con la identidad real
        
        self.modelo_voz = KNeighborsClassifier(n_neighbors=3)

    # ==========================================
    # GESTIÓN DE MODELOS EN DISCO
    # ==========================================
    def guardar_modelos(self, ruta="data/modelos_entrenados.pkl"):
        estado = {
            'modelo_vision': self.modelo_vision,
            'centroides': self.centroides,
            'modelo_voz': self.modelo_voz,
            'scaler_vision': self.scaler_vision,
            'mapa_clusters': self.mapa_clusters # <-- Guardamos el mapeo
        }
        with open(ruta, 'wb') as archivo:
            pickle.dump(estado, archivo)

    def cargar_modelos(self, ruta="data/modelos_entrenados.pkl"):
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró '{ruta}'. Entrena los modelos primero.")
        with open(ruta, 'rb') as archivo:
            estado = pickle.load(archivo)
        self.modelo_vision = estado['modelo_vision']
        self.centroides = estado['centroides']
        self.modelo_voz = estado['modelo_voz']
        self.scaler_vision = estado['scaler_vision']
        self.mapa_clusters = estado['mapa_clusters'] # <-- Cargamos el mapeo

    # ==========================================
    # MÓDULO DE VISIÓN (HUELLAS DACTILARES)
    # ==========================================
    def preprocesar_huella(self, ruta_o_imagen):
        return procesar_pipeline_completo(ruta_o_imagen)

    def extraer_caracteristicas_huella(self, imagen_esqueletizada):
        momentos = cv2.moments(imagen_esqueletizada)
        momentos_hu = cv2.HuMoments(momentos)
        caracteristicas = []
        for i in range(0, 7):
            if momentos_hu[i][0] != 0:
                caracteristicas.append(-1 * np.copysign(1.0, momentos_hu[i][0]) * np.log10(abs(momentos_hu[i][0])))
            else:
                caracteristicas.append(0)
        return np.array(caracteristicas)

    def entrenar_kmeans(self, directorio_autorizadas="data/huellas/autorizadas"):
        print("\nEntrenando modelo de visión (K-Means)...")
        X_vision = []
        etiquetas_reales = [] # <-- NUEVO: Guardamos de qué carpeta viene cada huella
        
        if not os.path.exists(directorio_autorizadas):
            print(f"Advertencia: No se encontró '{directorio_autorizadas}'.")
            return

        for carpeta_id in os.listdir(directorio_autorizadas):
            ruta_id = os.path.join(directorio_autorizadas, carpeta_id)
            if os.path.isdir(ruta_id):
                for archivo in os.listdir(ruta_id):
                    if archivo.endswith(('.png', '.jpg')):
                        ruta_imagen = os.path.join(ruta_id, archivo)
                        try:
                            esqueleto = self.preprocesar_huella(ruta_imagen)
                            caracs = self.extraer_caracteristicas_huella(esqueleto)
                            X_vision.append(caracs)
                            etiquetas_reales.append(carpeta_id) # Guardamos el nombre real
                        except Exception:
                            pass

        if len(X_vision) >= 3:
            X_vision = np.array(X_vision)
            X_vision_escalado = self.scaler_vision.fit_transform(X_vision)
            
            # Entrenamos el K-Means
            self.modelo_vision.fit(X_vision_escalado)
            self.centroides = self.modelo_vision.cluster_centers_
            
            # <-- NUEVO: Mapeo de Clusters a Identidades Reales
            predicciones_kmeans = self.modelo_vision.labels_
            self.mapa_clusters = {}
            for k in range(self.modelo_vision.n_clusters):
                # Vemos qué etiquetas reales cayeron dentro del cluster K
                etiquetas_en_cluster = [etiquetas_reales[i] for i in range(len(predicciones_kmeans)) if predicciones_kmeans[i] == k]
                if etiquetas_en_cluster:
                    # La identidad ganadora es la que más se repite en ese cluster
                    identidad_mayoritaria = Counter(etiquetas_en_cluster).most_common(1)[0][0]
                    self.mapa_clusters[k] = identidad_mayoritaria
                    
            print(f"Mapeo de identidades resuelto: {self.mapa_clusters}")
            print("¡Modelo de visión K-Means entrenado exitosamente!")
        else:
            print("Error: Se necesitan al menos 3 muestras para entrenar.")

    def evaluar_acceso(self, ruta_huella_prueba):
        if self.centroides is None:
            return False, "Modelo no entrenado"

        esqueleto = self.preprocesar_huella(ruta_huella_prueba)
        caracteristicas_crudas = self.extraer_caracteristicas_huella(esqueleto)
        
        caracs_escaladas = self.scaler_vision.transform([caracteristicas_crudas])
        
        id_cluster_predicho = self.modelo_vision.predict(caracs_escaladas)[0]
        centroide_correspondiente = self.centroides[id_cluster_predicho]
        
        # <-- NUEVO: Traducimos el número arbitrario del cluster a tu identidad real
        identidad_real = self.mapa_clusters.get(id_cluster_predicho, f"Cluster {id_cluster_predicho}")
        
        distancia = np.linalg.norm(caracs_escaladas[0] - centroide_correspondiente)
        print(f"Distancia de prueba -> Centroide ({identidad_real}): {distancia:.4f}")
        
        if distancia <= self.umbral_aceptacion:
            return True, f"Identidad: {identidad_real.upper()}"
        else:
            return False, "Desconocido (Intruso)"

    # ==========================================
    # MÓDULO DE VOZ (COMANDOS)
    # ==========================================
    def entrenar_knn(self):
        print("\nEntrenando modelo de voz (K-NN)...")
        X_voz, y_voz = cargar_dataset_voz()
        if len(X_voz) > 0:
            self.modelo_voz.fit(X_voz, y_voz)
            print("¡Modelo de voz K-NN entrenado exitosamente!")

    def escuchar_comando(self):
        caracteristicas = escuchar_audio_en_vivo()
        if caracteristicas is not None:
            prediccion = self.modelo_voz.predict([caracteristicas])[0]
            return prediccion.lower()
        return None