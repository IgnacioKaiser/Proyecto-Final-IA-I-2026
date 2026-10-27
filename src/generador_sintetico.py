import cv2
import numpy as np
import os
import random

def aplicar_rotacion(imagen, angulo_maximo=15):
    """Aplica una rotación aleatoria entre -angulo_maximo y +angulo_maximo grados."""
    h, w = imagen.shape
    angulo = random.uniform(-angulo_maximo, angulo_maximo)
    matriz_rotacion = cv2.getRotationMatrix2D((w/2, h/2), angulo, 1.0)
    # Borde en color blanco (255) asumiendo fondo blanco de la huella
    imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (w, h), borderValue=255)
    return imagen_rotada

def aplicar_traslacion(imagen, desplazamiento_maximo=15):
    """Desplaza la imagen aleatoriamente en los ejes X e Y."""
    h, w = imagen.shape
    tx = random.randint(-desplazamiento_maximo, desplazamiento_maximo)
    ty = random.randint(-desplazamiento_maximo, desplazamiento_maximo)
    matriz_traslacion = np.float32([[1, 0, tx], [0, 1, ty]])
    imagen_trasladada = cv2.warpAffine(imagen, matriz_traslacion, (w, h), borderValue=255)
    return imagen_trasladada

def aplicar_ruido(imagen, probabilidad=0.02):
    """Añade ruido de sal y pimienta simulando suciedad en el sensor[cite: 3]."""
    imagen_ruidosa = imagen.copy()
    h, w = imagen.shape
    
    # Cantidad de píxeles afectados
    num_pixeles = int(probabilidad * h * w)
    
    for _ in range(num_pixeles):
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        # 50% de probabilidad de sal (blanco) y 50% de pimienta (negro)
        imagen_ruidosa[y, x] = 255 if random.random() > 0.5 else 0
        
    return imagen_ruidosa

def aplicar_corte_parcial(imagen, proporcion_corte=0.2):
    """Simula un corte parcial de la huella tapando un borde con blanco[cite: 3]."""
    imagen_cortada = imagen.copy()
    h, w = imagen.shape
    
    # Elegir aleatoriamente qué borde cortar (0: Arriba, 1: Abajo, 2: Izquierda, 3: Derecha)
    borde = random.randint(0, 3)
    
    if borde == 0:
        imagen_cortada[0:int(h * proporcion_corte), :] = 255
    elif borde == 1:
        imagen_cortada[int(h * (1 - proporcion_corte)):h, :] = 255
    elif borde == 2:
        imagen_cortada[:, 0:int(w * proporcion_corte)] = 255
    else:
        imagen_cortada[:, int(w * (1 - proporcion_corte)):w] = 255
        
    return imagen_cortada

def generar_variantes(ruta_base, carpeta_destino, cantidad=50):
    """
    Toma una imagen base y genera 'cantidad' de variantes aplicando
    transformaciones sintéticas aleatorias para simular casos reales[cite: 3].
    """
    os.makedirs(carpeta_destino, exist_ok=True)
    imagen_base = cv2.imread(ruta_base, cv2.IMREAD_GRAYSCALE)
    
    if imagen_base is None:
        print(f"Error: No se encontró la imagen base en {ruta_base}")
        return

    nombre_base = os.path.basename(ruta_base).split('.')[0]

    for i in range(cantidad):
        variante = imagen_base.copy()
        
        # Aplicamos transformaciones aleatorias (pueden combinarse)
        if random.random() > 0.3:  # 70% de probabilidad de rotarse
            variante = aplicar_rotacion(variante)
        if random.random() > 0.3:  # 70% de probabilidad de trasladarse
            variante = aplicar_traslacion(variante)
        if random.random() > 0.4:  # 60% de probabilidad de tener ruido
            variante = aplicar_ruido(variante)
        if random.random() > 0.8:  # 20% de probabilidad de sufrir un corte
            variante = aplicar_corte_parcial(variante)
            
        ruta_salida = os.path.join(carpeta_destino, f"{nombre_base}_var_{i+1}.png")
        cv2.imwrite(ruta_salida, variante)
        
    print(f"✅ Se generaron {cantidad} variantes en '{carpeta_destino}' a partir de {nombre_base}.")

if __name__ == "__main__":
    print("=== GENERADOR DE IDENTIDADES SINTÉTICAS ===")
    
    # 1. Generar lote de identidades AUTORIZADAS (Entrenamiento K-Means)
    # Suponiendo que tienes tus 3 huellas base guardadas aquí:
    generar_variantes("data/huellas/base/autorizada_1.png", "data/huellas/autorizadas/id_1", cantidad=30)
    generar_variantes("data/huellas/base/autorizada_2.png", "data/huellas/autorizadas/id_2", cantidad=30)
    generar_variantes("data/huellas/base/autorizada_3.png", "data/huellas/autorizadas/id_3", cantidad=30)
    
    # 2. Generar lote de INTRUSOS (Identidades desconocidas)
    # Tomas huellas base que no pertenecen a las 3 autorizadas
    generar_variantes("data/huellas/base/intrusa_1.png", "data/huellas/intrusos/id_desconocida_1", cantidad=20)
    generar_variantes("data/huellas/base/intrusa_2.png", "data/huellas/intrusos/id_desconocida_2", cantidad=20)

    # 3. Generar muestras de PRUEBA (Validación final)
    # Se genera un lote extra de las autorizadas para la demostración en vivo
    generar_variantes("data/huellas/base/autorizada_1.png", "data/huellas/prueba/autorizadas", cantidad=10)
    generar_variantes("data/huellas/base/autorizada_2.png", "data/huellas/prueba/autorizadas", cantidad=10)