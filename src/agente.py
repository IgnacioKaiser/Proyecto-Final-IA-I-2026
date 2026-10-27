import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

# Importamos las funciones de nuestros otros módulos
from src.procesamiento_voz import cargar_dataset_voz, escuchar_audio_en_vivo
from src.procesamiento_imagen import procesar_pipeline_completo

class AgenteControlAcceso:
    def __init__(self, umbral_aceptacion=0.5):
        """
        Inicializa el agente, los clasificadores y el umbral empírico.
        """
        # Agrupamiento de huellas: K-Means con k=3 para las clases autorizadas
        self.modelo_vision = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.centroides = None
        self.umbral_aceptacion = umbral_aceptacion # Umbral justificado experimentalmente[cite: 4]
        
        # Reconocimiento de voz: K-NN[cite: 4]
        self.modelo_voz = KNeighborsClassifier(n_neighbors=3)

    # ==========================================
    # MÓDULO DE VISIÓN (HUELLAS DACTILARES)
    # ==========================================
    
    def preprocesar_huella(self, ruta_o_imagen):
        """
        Acondiciona la imagen sintética ejecutando el pipeline completo: 
        recorte, normalización, filtrado, binarización y esqueletizado.
        """
        return procesar_pipeline_completo(ruta_o_imagen)

    def extraer_caracteristicas_huella(self, imagen_esqueletizada):
        """
        Genera un vector discriminante basado en los Momentos de Hu, 
        que describen matemáticamente la estructura de la huella[cite: 4].
        """
        # Calculamos los momentos espaciales de la imagen
        momentos = cv2.moments(imagen_esqueletizada)
        
        # Calculamos los 7 Momentos de Hu (invariantes a rotación y traslación)
        momentos_hu = cv2.HuMoments(momentos)
        
        # Aplicamos una transformación logarítmica para estabilizar los valores
        # ya que los momentos de Hu suelen ser números extremadamente pequeños
        caracteristicas = []
        for i in range(0, 7):
            if momentos_hu[i][0] != 0:
                caracteristicas.append(-1 * np.copysign(1.0, momentos_hu[i][0]) * np.log10(abs(momentos_hu[i][0])))
            else:
                caracteristicas.append(0)
                
        return np.array(caracteristicas)

    def entrenar_kmeans(self, directorio_autorizadas="data/huellas/autorizadas"):
        """
        Extrae características de todas las huellas autorizadas y calcula los 
        centroides (k=3) a partir de los datos sintéticos[cite: 3, 4].
        """
        print("\nEntrenando modelo de visión (K-Means)...")
        X_vision = []
        
        if not os.path.exists(directorio_autorizadas):
            print(f"Advertencia: No se encontró la ruta '{directorio_autorizadas}'.")
            return

        # Recorremos las subcarpetas de identidades
        for carpeta_id in os.listdir(directorio_autorizadas):
            ruta_id = os.path.join(directorio_autorizadas, carpeta_id)
            if os.path.isdir(ruta_id):
                for archivo in os.listdir(ruta_id):
                    if archivo.endswith(('.png', '.jpg')):
                        ruta_imagen = os.path.join(ruta_id, archivo)
                        try:
                            esqueleto = self.preprocesar_huella(ruta_imagen)
                            caracteristicas = self.extraer_caracteristicas_huella(esqueleto)
                            X_vision.append(caracteristicas)
                        except Exception as e:
                            print(f"Error al procesar {ruta_imagen}: {e}")

        if len(X_vision) >= 3:
            # Entrenamos el modelo con la matriz de características extraídas
            self.modelo_vision.fit(X_vision)
            self.centroides = self.modelo_vision.cluster_centers_
            print("¡Modelo de visión K-Means entrenado exitosamente!")
        else:
            print("Error: Se necesitan al menos 3 muestras para entrenar K-Means con k=3.")

    def evaluar_acceso(self, ruta_huella_prueba):
        """
        Evalúa si una huella pertenece a una clase autorizada midiendo la 
        distancia Euclidiana al centroide más cercano[cite: 3].
        """
        if self.centroides is None:
            return False, "Modelo no entrenado"

        # 1. Preprocesar y extraer características de la nueva huella
        esqueleto = self.preprocesar_huella(ruta_huella_prueba)
        caracteristicas = self.extraer_caracteristicas_huella(esqueleto)
        
        # 2. Predecir a qué clúster (identidad) pertenece
        id_predicha = self.modelo_vision.predict([caracteristicas])[0]
        
        # 3. Obtener el centroide de ese clúster y calcular la distancia
        centroide_correspondiente = self.centroides[id_predicha]
        distancia = np.linalg.norm(caracteristicas - centroide_correspondiente)
        
        print(f"Distancia al centroide de la Identidad {id_predicha + 1}: {distancia:.4f}")
        
        # 4. Decisión basada en el umbral[cite: 4]
        if distancia <= self.umbral_aceptacion:
            return True, f"Identidad Sintética #{id_predicha + 1}"
        else:
            return False, "Desconocido (Intruso)"

    # ==========================================
    # MÓDULO DE VOZ (COMANDOS)
    # ==========================================
    
    def entrenar_knn(self):
        """
        Carga el dataset de audios y entrena el clasificador K-NN[cite: 4].
        """
        print("\nEntrenando modelo de voz (K-NN)...")
        X_voz, y_voz = cargar_dataset_voz()
        
        if len(X_voz) > 0:
            self.modelo_voz.fit(X_voz, y_voz)
            print("¡Modelo de voz K-NN entrenado exitosamente!")
        else:
            print("Advertencia: No se encontraron audios para entrenar K-NN.")

    def escuchar_comando(self):
        """
        Captura audio del micrófono, extrae características y clasifica 
        el comando diciendo "reconocer", "identificar" o "salir"[cite: 4].
        """
        caracteristicas = escuchar_audio_en_vivo()
        if caracteristicas is not None:
            prediccion = self.modelo_voz.predict([caracteristicas])[0]
            return prediccion.lower()
        return None

