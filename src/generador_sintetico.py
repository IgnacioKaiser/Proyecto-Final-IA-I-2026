import cv2
import numpy as np
import os
import random
import re
from pathlib import Path

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

def aplicar_ruido_gaussiano(imagen, desviacion=8):
    """Añade ruido gaussiano de baja intensidad."""
    ruido = np.random.normal(0, desviacion, imagen.shape)
    imagen_ruidosa = imagen.astype(np.float32) + ruido
    return np.clip(imagen_ruidosa, 0, 255).astype(np.uint8)

def variar_presion(imagen):
    """Simula cambios de grosor de crestas mediante morfología."""
    operacion = random.choice((cv2.MORPH_ERODE, cv2.MORPH_DILATE))
    kernel = np.ones((2, 2), dtype=np.uint8)
    return cv2.morphologyEx(imagen, operacion, kernel)

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

def generar_variantes(ruta_base, carpeta_destino, cantidad=5, semilla=42):
    """
    Toma una imagen base y genera 'cantidad' de variantes aplicando
    transformaciones sintéticas aleatorias para simular casos reales[cite: 3].
    """
    os.makedirs(carpeta_destino, exist_ok=True)
    imagen_base = cv2.imread(str(ruta_base), cv2.IMREAD_GRAYSCALE)

    if imagen_base is None:
        raise FileNotFoundError(f"No se encontró la imagen base en {ruta_base}")

    nombre_base = Path(ruta_base).stem
    generadas = 0

    for i in range(cantidad):
        ruta_salida = os.path.join(carpeta_destino, f"{nombre_base}_var_{i+1}.png")
        if os.path.isfile(ruta_salida):
            continue

        semilla_variante = semilla + i
        rng = random.Random(semilla_variante)
        rng_numpy = np.random.default_rng(semilla_variante)
        variante = imagen_base.copy()

        if rng.random() > 0.2:
            angulo = rng.uniform(-12, 12)
            alto, ancho = variante.shape
            matriz_rotacion = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo, 1)
            variante = cv2.warpAffine(
                variante,
                matriz_rotacion,
                (ancho, alto),
                borderValue=255,
            )
        if rng.random() > 0.2:
            alto, ancho = variante.shape
            tx = rng.randint(-max(1, ancho // 30), max(1, ancho // 30))
            ty = rng.randint(-max(1, alto // 30), max(1, alto // 30))
            matriz_traslacion = np.float32([[1, 0, tx], [0, 1, ty]])
            variante = cv2.warpAffine(
                variante,
                matriz_traslacion,
                (ancho, alto),
                borderValue=255,
            )

        perturbacion = rng.choice(("sal_pimienta", "gaussiano", "presion", "corte"))
        if perturbacion == "sal_pimienta":
            alto, ancho = variante.shape
            cantidad_pixeles = int(0.001 * alto * ancho)
            ys = rng_numpy.integers(0, alto, cantidad_pixeles)
            xs = rng_numpy.integers(0, ancho, cantidad_pixeles)
            valores = rng_numpy.choice((0, 255), cantidad_pixeles)
            variante[ys, xs] = valores
        elif perturbacion == "gaussiano":
            ruido = rng_numpy.normal(0, 5, variante.shape)
            variante = np.clip(variante.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
        elif perturbacion == "presion":
            operacion = rng.choice((cv2.MORPH_ERODE, cv2.MORPH_DILATE))
            variante = cv2.morphologyEx(
                variante,
                operacion,
                np.ones((2, 2), dtype=np.uint8),
            )
        else:
            alto, ancho = variante.shape
            proporcion = 0.1
            borde = rng.randrange(4)
            if borde == 0:
                variante[:int(alto * proporcion), :] = 255
            elif borde == 1:
                variante[int(alto * (1 - proporcion)):, :] = 255
            elif borde == 2:
                variante[:, :int(ancho * proporcion)] = 255
            else:
                variante[:, int(ancho * (1 - proporcion)):] = 255

        if not cv2.imwrite(ruta_salida, variante):
            raise OSError(f"No se pudo guardar la variante en '{ruta_salida}'.")
        generadas += 1

    return generadas


def generar_variantes_impresiones(
    directorio_autorizadas="data/huellas/autorizadas",
    cantidad_por_imagen=5,
):
    """Genera variantes sin modificar originales y las guarda por impresión de origen."""
    directorio_autorizadas = Path(directorio_autorizadas)
    if not directorio_autorizadas.is_dir():
        raise FileNotFoundError(
            f"No se encontró el directorio de impresiones '{directorio_autorizadas}'."
        )

    carpetas = sorted(
        (
            ruta
            for ruta in directorio_autorizadas.iterdir()
            if ruta.is_dir() and re.fullmatch(r"Impression_\d+", ruta.name)
        ),
        key=lambda ruta: int(ruta.name.split("_")[-1]),
    )
    if not carpetas:
        return 0

    directorio_variantes = directorio_autorizadas.parent / "variantes"
    generadas = 0
    for carpeta in carpetas:
        for identidad in range(1, 7):
            ruta_origen = carpeta / f"{identidad}.png"
            if not ruta_origen.is_file():
                raise FileNotFoundError(
                    f"Falta '{ruta_origen}'. Cada impresión debe contener 1.png a 6.png."
                )
            generadas += generar_variantes(
                ruta_origen,
                directorio_variantes / carpeta.name,
                cantidad=cantidad_por_imagen,
                semilla=int(carpeta.name.split("_")[-1]) * 1000 + identidad * 100,
            )

    print(
        f"Se generaron {generadas} variantes en '{directorio_variantes}'. "
        "Las capturas originales quedaron intactas."
    )
    return generadas

if __name__ == "__main__":
    generar_variantes_impresiones()