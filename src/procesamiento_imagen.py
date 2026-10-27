import cv2
import numpy as np
from skimage.morphology import skeletonize

def aplicar_recorte_roi(imagen):
    """
    Recorta la región de interés (ROI) para eliminar bordes vacíos y centrar la huella.
    """
    # Encontrar los píxeles que no son blancos (fondo)
    puntos_oscuros = cv2.findNonZero(cv2.bitwise_not(imagen))
    if puntos_oscuros is not None:
        x, y, w, h = cv2.boundingRect(puntos_oscuros)
        # Añadir un pequeño margen de seguridad
        margen = 10
        x = max(0, x - margen)
        y = max(0, y - margen)
        w = min(imagen.shape[1] - x, w + 2 * margen)
        h = min(imagen.shape[0] - y, h + 2 * margen)
        return imagen[y:y+h, x:x+w]
    return imagen

def normalizar_y_filtrar(imagen):
    """
    Mejora el contraste usando ecualización adaptativa (CLAHE) y aplica un filtro
    espacial de mediana para eliminar ruido (como sal y pimienta)[cite: 3].
    """
    # Normalización de contraste (CLAHE es ideal para resaltar crestas)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagen_contrastada = clahe.apply(imagen)
    
    # Filtrado espacial por mediana (suaviza preservando bordes)[cite: 3]
    imagen_filtrada = cv2.medianBlur(imagen_contrastada, 3)
    
    return imagen_filtrada

def binarizar_imagen(imagen):
    """
    Convierte la imagen a blanco y negro puro (0 y 255) usando el método de Otsu[cite: 3].
    """
    _, umbralizada = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return umbralizada

def aplicar_esqueletizado(imagen_binaria):
    """
    Reduce las crestas de la huella a un grosor de 1 píxel usando scikit-image[cite: 3].
    """
    # scikit-image requiere que la imagen sea booleana (True/False) o de valores 0/1
    imagen_booleana = imagen_binaria > 0
    
    # Aplicar el algoritmo de esqueletización[cite: 3]
    esqueleto = skeletonize(imagen_booleana)
    
    # Volver a convertir al formato de OpenCV (0 a 255, tipo uint8)
    esqueleto_cv2 = (esqueleto * 255).astype(np.uint8)
    return esqueleto_cv2

def procesar_pipeline_completo(ruta_o_imagen):
    """
    Ejecuta el flujo completo de acondicionamiento sobre una huella[cite: 3].
    Acepta tanto la ruta del archivo como la matriz de la imagen ya cargada.
    """
    # 1. Carga de la imagen en escala de grises
    if isinstance(ruta_o_imagen, str):
        imagen = cv2.imread(ruta_o_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen en: {ruta_o_imagen}")
    else:
        imagen = ruta_o_imagen.copy()
        if len(imagen.shape) == 3:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # 2. Recorte de ROI[cite: 3]
    imagen_roi = aplicar_recorte_roi(imagen)
    
    # 3. Normalización de contraste y filtrado espacial[cite: 3]
    imagen_limpia = normalizar_y_filtrar(imagen_roi)
    
    # 4. Binarización[cite: 3]
    imagen_binaria = binarizar_imagen(imagen_limpia)
    
    # 5. Esqueletizado[cite: 3]
    huella_esqueletizada = aplicar_esqueletizado(imagen_binaria)
    
    return huella_esqueletizada