import numpy as np
import cv2
import os

def generar_huella_procedural(tipo, semilla, ruta_salida):
    """
    Genera matemáticamente el patrón de una huella dactilar desde cero.
    """
    # Fijar la semilla para que la huella sea siempre idéntica para esa semilla
    np.random.seed(semilla)
    
    # Dimensiones de la huella
    ancho, alto = 300, 400
    Y, X = np.ogrid[:alto, :ancho]
    cx, cy = ancho // 2, alto // 2  # Centro
    
    # 1. Generar ruido suavizado para distorsionar la perfección matemática
    # Esto simula las imperfecciones y ondulaciones biológicas de la piel
    ruido = np.random.normal(0, 15, (alto, ancho))
    ruido_suavizado = cv2.GaussianBlur(ruido, (31, 31), 10) * 10
    
    # 2. Definir la topología geométrica (Fase matemática)
    if tipo == 'espiral':
        # Ecuación circular: (x-cx)^2 + (y-cy)^2
        distancia = np.sqrt((X - cx)**2 + (Y - cy)**2)
        fase = distancia + ruido_suavizado
        
    elif tipo == 'arco':
        # Ecuación parabólica suave
        distancia = Y + ((X - cx)**2) / 150.0
        fase = distancia + ruido_suavizado
        
    elif tipo == 'bucle':
        # Ecuación asimétrica desplazada
        distancia = np.sqrt((X - cx + 50)**2 + (Y - cy - 20)**2) + 0.6 * Y
        fase = distancia + ruido_suavizado
        
    else:
        raise ValueError("Tipo de huella no soportado.")

    # 3. Generar las crestas usando una función trigonométrica (Seno)
    frecuencia = 0.5  # Define el grosor de las líneas (crestas)
    patron = np.sin(frecuencia * fase)
    
    # 4. Binarizar: Si el valor de la onda es positivo es negro (0), si no, blanco (255)
    huella_binaria = np.where(patron > 0, 0, 255).astype(np.uint8)
    
    # 5. Aplicar una máscara elíptica para darle forma de dedo impreso
    mascara = np.zeros((alto, ancho), dtype=np.float32)
    cv2.ellipse(mascara, (cx, cy), (ancho//2 - 30, alto//2 - 40), 0, 0, 360, 1.0, -1)
    # Difuminar los bordes de la máscara
    mascara_suave = cv2.GaussianBlur(mascara, (51, 51), 20)
    
    # Mezclar la huella con un fondo completamente blanco
    huella_final = (huella_binaria * mascara_suave + 255 * (1 - mascara_suave)).astype(np.uint8)
    
    # Asegurar que sea blanco y negro puro (para evitar grises en los bordes)
    _, huella_final = cv2.threshold(huella_final, 127, 255, cv2.THRESH_BINARY)

    # 6. Guardar la imagen generada
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    cv2.imwrite(ruta_salida, huella_final)
    print(f"✅ Huella base tipo '{tipo.upper()}' (Semilla: {semilla}) generada en: {ruta_salida}")

if __name__ == "__main__":
    print("=== GENERADOR PROCEDURAL DE HUELLAS BASE ===")
    
    # Generar 3 identidades autorizadas base
    # Al cambiar el número de semilla, cambia la forma biológica exacta de la huella
    generar_huella_procedural('arco', semilla=101, ruta_salida='data/huellas/base/autorizada_1.png')
    generar_huella_procedural('bucle', semilla=202, ruta_salida='data/huellas/base/autorizada_2.png')
    generar_huella_procedural('espiral', semilla=303, ruta_salida='data/huellas/base/autorizada_3.png')
    
    # Generar 2 identidades intrusas base
    generar_huella_procedural('arco', semilla=999, ruta_salida='data/huellas/base/intrusa_1.png')
    generar_huella_procedural('espiral', semilla=888, ruta_salida='data/huellas/base/intrusa_2.png')