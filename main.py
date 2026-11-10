import random
import sys
from pathlib import Path

import pygame

from src.agente import AgenteControlAcceso, obtener_particiones_impresiones


DIRECTORIO_PROYECTO = Path(__file__).resolve().parent
DIRECTORIO_PRUEBA = DIRECTORIO_PROYECTO / "data" / "huellas" / "prueba"
DIRECTORIO_INTRUSOS = DIRECTORIO_PROYECTO / "data" / "huellas" / "intrusos"

ANCHO, ALTO = 800, 600
COLOR_FONDO = (30, 30, 30)
COLOR_TEXTO = (240, 240, 240)
COLOR_BOTON = (45, 105, 170)
COLOR_BOTON_HOVER = (65, 130, 200)
COLOR_EXITO = (0, 180, 50)
COLOR_ERROR = (220, 55, 55)

BOTON_ESCUCHAR = pygame.Rect(65, 510, 310, 58)
BOTON_OTRA_HUELLA = pygame.Rect(425, 510, 310, 58)


def obtener_muestras_prueba():
    """Reúne las huellas autorizadas de prueba y las muestras de intrusos."""
    particiones = obtener_particiones_impresiones(
        DIRECTORIO_PROYECTO / "data" / "huellas" / "autorizadas"
    )
    if particiones:
        return sorted({ruta for ruta, _ in particiones["prueba"]})

    muestras = []
    for directorio in (DIRECTORIO_PRUEBA, DIRECTORIO_INTRUSOS):
        if directorio.is_dir():
            muestras.extend(
                ruta
                for ruta in directorio.rglob("*")
                if ruta.is_file() and ruta.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )

    muestras.sort()
    if not muestras:
        raise FileNotFoundError(
            "No se encontraron huellas de prueba en data/huellas/prueba ni data/huellas/intrusos."
        )
    return muestras


def elegir_otra_muestra(muestras, muestra_actual=None):
    """Elige una huella al azar y evita repetir la actual cuando sea posible."""
    opciones = [muestra for muestra in muestras if muestra != muestra_actual]
    return random.choice(opciones or muestras)


def dibujar_boton(pantalla, fuente, rectangulo, etiqueta, posicion_mouse):
    color = COLOR_BOTON_HOVER if rectangulo.collidepoint(posicion_mouse) else COLOR_BOTON
    pygame.draw.rect(pantalla, color, rectangulo, border_radius=8)
    texto = fuente.render(etiqueta, True, COLOR_TEXTO)
    pantalla.blit(
        texto,
        (rectangulo.centerx - texto.get_width() // 2, rectangulo.centery - texto.get_height() // 2),
    )


def dibujar_interfaz(pantalla, fuentes, ruta_muestra, imagen_muestra, mensaje, color_mensaje):
    fuente_titulo, fuente_texto, fuente_boton = fuentes
    pantalla.fill(COLOR_FONDO)

    titulo = fuente_titulo.render("CONTROL DE ACCESO", True, COLOR_TEXTO)
    pantalla.blit(titulo, (ANCHO // 2 - titulo.get_width() // 2, 28))

    pygame.draw.rect(pantalla, (55, 55, 55), (280, 100, 240, 250), border_radius=8)
    pantalla.blit(
        imagen_muestra,
        (
            ANCHO // 2 - imagen_muestra.get_width() // 2,
            225 - imagen_muestra.get_height() // 2,
        ),
    )

    try:
        nombre_muestra = str(
            ruta_muestra.relative_to(
                DIRECTORIO_PROYECTO / "data" / "huellas" / "autorizadas"
            )
        )
    except ValueError:
        nombre_muestra = ruta_muestra.name
    etiqueta_muestra = fuente_texto.render(nombre_muestra, True, COLOR_TEXTO)
    pantalla.blit(
        etiqueta_muestra,
        (ANCHO // 2 - etiqueta_muestra.get_width() // 2, 365),
    )

    texto_estado = fuente_texto.render(mensaje, True, color_mensaje)
    pantalla.blit(texto_estado, (ANCHO // 2 - texto_estado.get_width() // 2, 430))

    posicion_mouse = pygame.mouse.get_pos()
    dibujar_boton(
        pantalla,
        fuente_boton,
        BOTON_ESCUCHAR,
        "ESCUCHAR COMANDO",
        posicion_mouse,
    )
    dibujar_boton(
        pantalla,
        fuente_boton,
        BOTON_OTRA_HUELLA,
        "OTRA HUELLA ALEATORIA",
        posicion_mouse,
    )
    pygame.display.flip()


def cargar_imagen_muestra(ruta_muestra):
    imagen = pygame.image.load(str(ruta_muestra)).convert()
    ancho, alto = imagen.get_size()
    escala = min(220 / ancho, 230 / alto)
    nuevo_tamano = (max(1, int(ancho * escala)), max(1, int(alto * escala)))
    return pygame.transform.smoothscale(imagen, nuevo_tamano)


def main():
    print("=== INICIANDO SISTEMA MULTIMODAL ===")
    agente = AgenteControlAcceso()
    agente.entrenar_kmeans()
    agente.entrenar_knn()
    print(f"Calibración del umbral: {agente.calibrar_umbral()}")

    muestras = obtener_muestras_prueba()
    ruta_muestra = elegir_otra_muestra(muestras)

    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Agente de Control de Acceso Multimodal")
    fuentes = (
        pygame.font.SysFont("consolas", 34, bold=True),
        pygame.font.SysFont("consolas", 19),
        pygame.font.SysFont("consolas", 20, bold=True),
    )

    imagen_muestra = cargar_imagen_muestra(ruta_muestra)
    mensaje = "Huella seleccionada. Presione un botón."
    color_mensaje = COLOR_TEXTO
    reloj = pygame.time.Clock()

    ejecutando = True
    while ejecutando:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                ejecutando = False
            elif evento.type == pygame.KEYDOWN and evento.key == pygame.K_ESCAPE:
                ejecutando = False
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                if BOTON_OTRA_HUELLA.collidepoint(evento.pos):
                    ruta_muestra = elegir_otra_muestra(muestras, ruta_muestra)
                    imagen_muestra = cargar_imagen_muestra(ruta_muestra)
                    mensaje = "Nueva huella seleccionada."
                    color_mensaje = COLOR_TEXTO
                elif BOTON_ESCUCHAR.collidepoint(evento.pos):
                    mensaje = "Escuchando: diga reconocer, identificar o salir..."
                    color_mensaje = COLOR_TEXTO
                    dibujar_interfaz(
                        pantalla,
                        fuentes,
                        ruta_muestra,
                        imagen_muestra,
                        mensaje,
                        color_mensaje,
                    )

                    comando = agente.escuchar_comando()
                    print(f"Comando detectado: {comando if comando is not None else 'ninguno'}")
                    if comando == "salir":
                        ejecutando = False
                    elif comando in {"reconocer", "identificar"}:
                        acceso, identidad = agente.evaluar_acceso(str(ruta_muestra))
                        if acceso and comando == "identificar":
                            mensaje = f"IDENTIDAD RECONOCIDA: {identidad}"
                            color_mensaje = COLOR_EXITO
                        elif acceso:
                            mensaje = "HUELLA AUTORIZADA"
                            color_mensaje = COLOR_EXITO
                        else:
                            mensaje = "HUELLA NO AUTORIZADA"
                            color_mensaje = COLOR_ERROR
                    elif comando is None:
                        mensaje = "No se detecto ningun comando. Intente nuevamente."
                        color_mensaje = COLOR_ERROR
                    else:
                        mensaje = f"Comando no reconocido: {comando}"
                        color_mensaje = COLOR_ERROR

        dibujar_interfaz(
            pantalla,
            fuentes,
            ruta_muestra,
            imagen_muestra,
            mensaje,
            color_mensaje,
        )
        reloj.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
