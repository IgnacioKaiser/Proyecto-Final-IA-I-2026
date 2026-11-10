import numpy as np
import cv2
import os
import pickle
import re
from pathlib import Path
from itertools import permutations
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from collections import Counter

from src.procesamiento_voz import cargar_dataset_voz, escuchar_audio_en_vivo
from src.procesamiento_imagen import procesar_pipeline_completo
from src.generador_sintetico import generar_variantes_impresiones


def obtener_particiones_impresiones(directorio_autorizadas="data/huellas/autorizadas"):
    """Separa las carpetas de impresiones en entrenamiento, calibración y prueba."""
    directorio = Path(directorio_autorizadas)
    carpetas = [
        (int(coincidencia.group(1)), ruta)
        for ruta in directorio.iterdir()
        if ruta.is_dir()
        and (coincidencia := re.fullmatch(r"Impression_(\d+)", ruta.name))
    ] if directorio.is_dir() else []
    if not carpetas:
        return None
    if len(carpetas) < 5:
        raise ValueError("Se necesitan al menos 5 carpetas Impression_N para separar los conjuntos.")

    carpetas.sort()
    muestras_por_carpeta = {}
    for numero, carpeta in carpetas:
        archivos = {
            Path(nombre).stem: carpeta / nombre
            for nombre in os.listdir(carpeta)
            if (carpeta / nombre).is_file()
            and Path(nombre).suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        faltantes = [str(identidad) for identidad in range(1, 7) if str(identidad) not in archivos]
        if faltantes:
            raise ValueError(
                f"Faltan las imágenes {', '.join(faltantes)} en '{carpeta}'. "
                "Cada impresión debe contener 1.png a 6.png."
            )
        muestras = []
        directorio_variantes = directorio.parent / "variantes" / carpeta.name
        for identidad in range(1, 7):
            muestras.append((archivos[str(identidad)], identidad))
            variantes = sorted(
                directorio_variantes.glob(f"{identidad}_var_*.png"),
                key=lambda ruta: int(ruta.stem.rsplit("_", 1)[-1]),
            )
            muestras.extend((ruta, identidad) for ruta in variantes)
        muestras_por_carpeta[numero] = muestras

    orden = np.random.default_rng(42).permutation(len(carpetas))
    cantidad_entrenamiento = max(1, int(len(carpetas) * 0.6))
    cantidad_calibracion = max(1, int(len(carpetas) * 0.2))
    if cantidad_entrenamiento + cantidad_calibracion >= len(carpetas):
        cantidad_entrenamiento = len(carpetas) - 2
        cantidad_calibracion = 1

    carpetas_entrenamiento = {carpetas[indice][0] for indice in orden[:cantidad_entrenamiento]}
    carpetas_calibracion = {
        carpetas[indice][0]
        for indice in orden[cantidad_entrenamiento:cantidad_entrenamiento + cantidad_calibracion]
    }
    return {
        "entrenamiento": [
            muestra
            for numero, muestras in muestras_por_carpeta.items()
            if numero in carpetas_entrenamiento
            for muestra in muestras
            if muestra[1] <= 3
        ],
        "calibracion": [
            muestra
            for numero, muestras in muestras_por_carpeta.items()
            if numero in carpetas_calibracion
            for muestra in muestras
        ],
        "prueba": [
            muestra
            for numero, muestras in muestras_por_carpeta.items()
            if numero not in carpetas_entrenamiento and numero not in carpetas_calibracion
            for muestra in muestras
        ],
    }


class AgenteControlAcceso:
    def __init__(self, umbral_aceptacion=1.5): # <-- Umbral más estricto por la normalización
        self.modelo_vision = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.centroides = None
        self.umbral_aceptacion = umbral_aceptacion
        self.scaler_vision = StandardScaler()
        self.mapa_clusters = {} # <-- NUEVO: Diccionario para vincular K-Means con la identidad real
        
        self.modelo_voz = make_pipeline(
            RobustScaler(),
            KNeighborsClassifier(n_neighbors=3),
        )
        self.umbral_distancia_voz = None
        self.confianza_minima_voz = 2 / 3

    # ==========================================
    # GESTIÓN DE MODELOS EN DISCO
    # ==========================================
    def guardar_modelos(self, ruta="data/modelos_entrenados.pkl"):
        estado = {
            'modelo_vision': self.modelo_vision,
            'centroides': self.centroides,
            'modelo_voz': self.modelo_voz,
            'scaler_vision': self.scaler_vision,
            'mapa_clusters': self.mapa_clusters,
            'umbral_aceptacion': self.umbral_aceptacion,
            'umbral_distancia_voz': self.umbral_distancia_voz,
            'confianza_minima_voz': self.confianza_minima_voz,
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
        self.umbral_aceptacion = estado.get('umbral_aceptacion', self.umbral_aceptacion)
        self.umbral_distancia_voz = estado.get('umbral_distancia_voz')
        self.confianza_minima_voz = estado.get('confianza_minima_voz', self.confianza_minima_voz)

    # ==========================================
    # MÓDULO DE VISIÓN (HUELLAS DACTILARES)
    # ==========================================
    def preprocesar_huella(self, ruta_o_imagen):
        return procesar_pipeline_completo(ruta_o_imagen)

    def extraer_caracteristicas_huella(self, imagen_esqueletizada):
        imagen_normalizada = cv2.resize(
            imagen_esqueletizada,
            (128, 128),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255
        return hog(
            imagen_normalizada,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            feature_vector=True,
        ).astype(np.float64)

    def entrenar_kmeans(self, directorio_autorizadas="data/huellas/autorizadas"):
        print("\nEntrenando modelo de visión (K-Means)...")
        X_vision = []
        etiquetas_reales = []
        
        if not os.path.exists(directorio_autorizadas):
            print(f"Advertencia: No se encontró '{directorio_autorizadas}'.")
            return

        if obtener_particiones_impresiones(directorio_autorizadas):
            generar_variantes_impresiones(directorio_autorizadas)
        particiones = obtener_particiones_impresiones(directorio_autorizadas)
        if particiones:
            muestras_entrenamiento = [
                (ruta, f"id_{identidad}")
                for ruta, identidad in particiones["entrenamiento"]
            ]
        else:
            muestras_entrenamiento = []
            for carpeta_id in os.listdir(directorio_autorizadas):
                ruta_id = os.path.join(directorio_autorizadas, carpeta_id)
                if os.path.isdir(ruta_id):
                    for archivo in os.listdir(ruta_id):
                        if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
                            muestras_entrenamiento.append(
                                (Path(ruta_id) / archivo, carpeta_id)
                            )

        for ruta_imagen, etiqueta in muestras_entrenamiento:
            esqueleto = self.preprocesar_huella(str(ruta_imagen))
            caracs = self.extraer_caracteristicas_huella(esqueleto)
            X_vision.append(caracs)
            etiquetas_reales.append(etiqueta)

        if len(X_vision) >= 3:
            X_vision = np.array(X_vision)
            X_vision_escalado = self.scaler_vision.fit_transform(X_vision)
            
            # Entrenamos el K-Means
            self.modelo_vision.fit(X_vision_escalado)
            self.centroides = self.modelo_vision.cluster_centers_
            
            # <-- NUEVO: Mapeo de Clusters a Identidades Reales
            predicciones_kmeans = self.modelo_vision.labels_
            self.mapa_clusters = {}
            identidades = sorted(set(etiquetas_reales))
            if len(identidades) == self.modelo_vision.n_clusters:
                conteos = {
                    (cluster, identidad): sum(
                        etiqueta == identidad and prediccion == cluster
                        for etiqueta, prediccion in zip(etiquetas_reales, predicciones_kmeans)
                    )
                    for cluster in range(self.modelo_vision.n_clusters)
                    for identidad in identidades
                }
                asignaciones = permutations(identidades)
                mejor_asignacion = max(
                    asignaciones,
                    key=lambda asignacion: sum(
                        conteos[(cluster, identidad)]
                        for cluster, identidad in enumerate(asignacion)
                    ),
                )
                self.mapa_clusters = dict(enumerate(mejor_asignacion))
            else:
                for k in range(self.modelo_vision.n_clusters):
                    etiquetas_en_cluster = [
                        etiquetas_reales[i]
                        for i in range(len(predicciones_kmeans))
                        if predicciones_kmeans[i] == k
                    ]
                    if etiquetas_en_cluster:
                        self.mapa_clusters[k] = Counter(etiquetas_en_cluster).most_common(1)[0][0]
                    
            print(f"Mapeo de identidades resuelto: {self.mapa_clusters}")
            print(f"¡Modelo de visión K-Means entrenado con {len(X_vision)} muestras!")
        else:
            raise ValueError("Se necesitan al menos 3 muestras autorizadas para entrenar K-Means.")

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

    def _distancias_a_centroides(self, directorio):
        if self.centroides is None:
            raise RuntimeError("Entrena o carga el modelo K-Means antes de calibrar.")
        if not os.path.isdir(directorio):
            raise FileNotFoundError(f"No se encontró el directorio de evaluación '{directorio}'.")

        rutas = []
        for raiz, _, archivos in os.walk(directorio):
            rutas.extend(
                os.path.join(raiz, archivo)
                for archivo in archivos
                if archivo.lower().endswith(('.png', '.jpg', '.jpeg'))
            )
        rutas.sort()
        if not rutas:
            raise ValueError(f"No hay imágenes de huellas en '{directorio}'.")

        return self._distancias_a_rutas(rutas)

    def _distancias_a_rutas(self, rutas):
        caracteristicas = []
        for ruta in rutas:
            esqueleto = self.preprocesar_huella(str(ruta))
            caracteristicas.append(self.extraer_caracteristicas_huella(esqueleto))

        escaladas = self.scaler_vision.transform(caracteristicas)
        distancias = np.linalg.norm(
            escaladas[:, np.newaxis, :] - self.centroides[np.newaxis, :, :],
            axis=2,
        )
        return np.min(distancias, axis=1)

    def calibrar_umbral(
        self,
        directorio_autorizadas="data/huellas/prueba/autorizadas",
        directorio_intrusos="data/huellas/intrusos",
        especificidad_minima=0.8,
    ):
        """Calibra con impresiones separadas por captura y reporta métricas en un conjunto no usado."""
        if not 0 < especificidad_minima <= 1:
            raise ValueError("La especificidad mínima debe ser mayor que 0 y como máximo 1.")

        particiones = obtener_particiones_impresiones()
        if particiones:
            muestras_calibracion = particiones["calibracion"]
            muestras_validacion = particiones["prueba"]
            distancias_calibracion = self._distancias_a_rutas(
                [ruta for ruta, _ in muestras_calibracion]
            )
            etiquetas_calibracion = np.array(
                [identidad <= 3 for _, identidad in muestras_calibracion],
                dtype=bool,
            )
            distancias_validacion = self._distancias_a_rutas(
                [ruta for ruta, _ in muestras_validacion]
            )
            etiquetas_validacion = np.array(
                [identidad <= 3 for _, identidad in muestras_validacion],
                dtype=bool,
            )
        else:
            distancias_autorizadas = self._distancias_a_centroides(directorio_autorizadas)
            distancias_intrusos = self._distancias_a_centroides(directorio_intrusos)
            indices_autorizadas = np.arange(len(distancias_autorizadas))
            indices_intrusos = np.arange(len(distancias_intrusos))
            mascara_validacion_autorizadas = indices_autorizadas % 3 == 0
            mascara_validacion_intrusos = indices_intrusos % 3 == 0
            if (
                np.count_nonzero(mascara_validacion_autorizadas) == 0
                or np.count_nonzero(mascara_validacion_intrusos) == 0
                or np.count_nonzero(~mascara_validacion_autorizadas) == 0
                or np.count_nonzero(~mascara_validacion_intrusos) == 0
            ):
                raise ValueError("Se requieren al menos 3 muestras de autorizados e intrusos para calibrar.")

            distancias_calibracion = np.concatenate((
                distancias_autorizadas[~mascara_validacion_autorizadas],
                distancias_intrusos[~mascara_validacion_intrusos],
            ))
            etiquetas_calibracion = np.concatenate((
                np.ones(np.count_nonzero(~mascara_validacion_autorizadas), dtype=bool),
                np.zeros(np.count_nonzero(~mascara_validacion_intrusos), dtype=bool),
            ))
            distancias_validacion = np.concatenate((
                distancias_autorizadas[mascara_validacion_autorizadas],
                distancias_intrusos[mascara_validacion_intrusos],
            ))
            etiquetas_validacion = np.concatenate((
                np.ones(np.count_nonzero(mascara_validacion_autorizadas), dtype=bool),
                np.zeros(np.count_nonzero(mascara_validacion_intrusos), dtype=bool),
            ))

        if not np.any(etiquetas_calibracion) or np.all(etiquetas_calibracion):
            raise ValueError("La calibración debe incluir autorizados e intrusos.")
        if not np.any(etiquetas_validacion) or np.all(etiquetas_validacion):
            raise ValueError("La validación debe incluir autorizados e intrusos.")

        valores = np.unique(distancias_calibracion)
        candidatos = np.concatenate((
            [valores[0] - 1e-9],
            (valores[:-1] + valores[1:]) / 2,
            [valores[-1]],
        ))
        opciones = []
        opciones_con_especificidad_minima = []
        for umbral in candidatos:
            predicciones = distancias_calibracion <= umbral
            sensibilidad = np.mean(predicciones[etiquetas_calibracion])
            especificidad = np.mean(~predicciones[~etiquetas_calibracion])
            balanceada = (sensibilidad + especificidad) / 2
            opcion = (balanceada, sensibilidad, especificidad, -umbral, umbral)
            opciones.append(opcion)
            if especificidad >= especificidad_minima:
                opciones_con_especificidad_minima.append(opcion)

        if opciones_con_especificidad_minima:
            mejor = max(
                opciones_con_especificidad_minima,
                key=lambda opcion: (opcion[1], opcion[2], opcion[3]),
            )
            objetivo_especificidad_alcanzado = True
        else:
            mejor = max(opciones)
            objetivo_especificidad_alcanzado = False

        self.umbral_aceptacion = float(mejor[4])
        predicciones = distancias_validacion <= self.umbral_aceptacion
        vp = np.count_nonzero(predicciones & etiquetas_validacion)
        fn = np.count_nonzero(~predicciones & etiquetas_validacion)
        vn = np.count_nonzero(~predicciones & ~etiquetas_validacion)
        fp = np.count_nonzero(predicciones & ~etiquetas_validacion)
        sensibilidad = vp / np.count_nonzero(etiquetas_validacion)
        especificidad = vn / np.count_nonzero(~etiquetas_validacion)
        total = len(distancias_validacion)
        return {
            "umbral": float(self.umbral_aceptacion),
            "especificidad_minima_calibracion": float(especificidad_minima),
            "objetivo_especificidad_alcanzado": objetivo_especificidad_alcanzado,
            "muestras_calibracion": int(len(distancias_calibracion)),
            "muestras_validacion": int(total),
            "verdaderos_positivos": int(vp),
            "falsos_negativos": int(fn),
            "verdaderos_negativos": int(vn),
            "falsos_positivos": int(fp),
            "exactitud": float((vp + vn) / total),
            "sensibilidad": float(sensibilidad),
            "especificidad": float(especificidad),
            "exactitud_balanceada": float((sensibilidad + especificidad) / 2),
        }

    # ==========================================
    # MÓDULO DE VOZ (COMANDOS)
    # ==========================================
    def entrenar_knn(self):
        print("\nEntrenando modelo de voz (K-NN)...")
        X_voz, y_voz = cargar_dataset_voz()
        cantidad_por_comando = Counter(y_voz)
        comandos_requeridos = {"reconocer", "identificar", "salir"}
        comandos_faltantes = comandos_requeridos - set(cantidad_por_comando)
        if comandos_faltantes:
            faltantes = ", ".join(sorted(comandos_faltantes))
            raise ValueError(f"Faltan muestras de voz para entrenar estos comandos: {faltantes}.")
        if min(cantidad_por_comando.values()) < 3:
            raise ValueError("Se necesitan al menos 3 muestras por comando para entrenar el K-NN.")

        validacion = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        distancias_validacion = np.empty(len(y_voz), dtype=float)
        predicciones_validacion = np.empty(len(y_voz), dtype=y_voz.dtype)
        confianzas_validacion = np.empty(len(y_voz), dtype=float)
        for indices_entrenamiento, indices_validacion in validacion.split(X_voz, y_voz):
            modelo = clone(self.modelo_voz)
            modelo.fit(X_voz[indices_entrenamiento], y_voz[indices_entrenamiento])
            caracteristicas_validacion = X_voz[indices_validacion]
            predicciones_validacion[indices_validacion] = modelo.predict(caracteristicas_validacion)
            confianzas_validacion[indices_validacion] = modelo.predict_proba(
                caracteristicas_validacion
            ).max(axis=1)
            caracteristicas_escaladas = modelo[:-1].transform(caracteristicas_validacion)
            distancias_validacion[indices_validacion] = modelo[-1].kneighbors(
                caracteristicas_escaladas,
                n_neighbors=3,
            )[0][:, -1]

        self.umbral_distancia_voz = float(np.quantile(distancias_validacion, 0.995))
        predicciones_aceptadas = (
            (confianzas_validacion >= self.confianza_minima_voz)
            & (distancias_validacion <= self.umbral_distancia_voz)
        )
        cobertura = np.mean(predicciones_aceptadas)
        if np.any(predicciones_aceptadas):
            exactitud_aceptadas = np.mean(
                predicciones_validacion[predicciones_aceptadas]
                == y_voz[predicciones_aceptadas]
            )
        else:
            exactitud_aceptadas = 0.0

        self.modelo_voz.fit(X_voz, y_voz)
        print(
            "Validación de voz con rechazo: "
            f"cobertura={cobertura:.1%}, "
            f"exactitud de aceptados={exactitud_aceptadas:.1%}, "
            f"distancia máxima={self.umbral_distancia_voz:.3f}"
        )
        print(f"Muestras de voz por comando: {dict(cantidad_por_comando)}")
        print("¡Modelo de voz K-NN entrenado exitosamente!")

    def escuchar_comando(self):
        caracteristicas = escuchar_audio_en_vivo()
        if caracteristicas is None or self.umbral_distancia_voz is None:
            return None

        caracteristicas = np.asarray(caracteristicas).reshape(1, -1)
        probabilidades = self.modelo_voz.predict_proba(caracteristicas)[0]
        indice_prediccion = int(np.argmax(probabilidades))
        if probabilidades[indice_prediccion] < self.confianza_minima_voz:
            return None

        caracteristicas_escaladas = self.modelo_voz[:-1].transform(caracteristicas)
        distancias = self.modelo_voz[-1].kneighbors(
            caracteristicas_escaladas,
            n_neighbors=3,
        )[0][0]
        if distancias[-1] > self.umbral_distancia_voz:
            return None

        return str(self.modelo_voz[-1].classes_[indice_prediccion]).lower()