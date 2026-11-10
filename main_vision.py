import pygame
import random
import sys
from pathlib import Path
from src.agente import AgenteControlAcceso, obtener_particiones_impresiones

DIRECTORIO_PROYECTO = Path(__file__).resolve().parent

# ==========================================
# CONFIGURACIÓN DE INTERFAZ (PYGAME)
# ==========================================
ANCHO, ALTO = 800, 600
COLOR_FONDO_NORMAL = (30, 30, 30)       # Gris oscuro (Dark mode)
COLOR_EXITO = (0, 180, 50)              # Verde
COLOR_ERROR = (200, 0, 0)               # Rojo
COLOR_TEXTO = (240, 240, 240)           # Blanco suave

def obtener_ruta_prueba(tecla, muestras_por_identidad):
    """
    Selecciona una captura de prueba de la identidad indicada por la tecla.
    """
    teclas = {
        pygame.K_1: 1,
        pygame.K_2: 2,
        pygame.K_3: 3,
        pygame.K_4: 4,
        pygame.K_5: 5,
        pygame.K_6: 6,
    }
    identidad = teclas.get(tecla)
    muestras = muestras_por_identidad.get(identidad, [])
    if muestras:
        return random.choice(muestras)
    if identidad in {1, 2, 3}:
        return DIRECTORIO_PROYECTO / "data" / "huellas" / "prueba" / "autorizadas" / f"autorizada_{identidad}_var_1.png"
    if identidad == 4:
        return DIRECTORIO_PROYECTO / "data" / "huellas" / "intrusos" / "id_desconocida_1" / "intrusa_1_var_1.png"
    return None

def main():
    print("=== INICIANDO MODO PRUEBA: SOLO VISIÓN ===")
    
    # 1. Instanciar agente y cargar el "cerebro" pre-entrenado
    agente = AgenteControlAcceso()
    agente.cargar_modelos(DIRECTORIO_PROYECTO / "data/modelos_entrenados.pkl")
    print(f"Calibración del umbral: {agente.calibrar_umbral()}")
    particiones = obtener_particiones_impresiones(
        DIRECTORIO_PROYECTO / "data" / "huellas" / "autorizadas"
    )
    muestras_por_identidad = {}
    if particiones:
        for ruta, identidad in particiones["prueba"]:
            muestras_por_identidad.setdefault(identidad, []).append(ruta)

    # 2. Inicializar el motor gráfico
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Test de Visión - Agente IA")
    
    fuente_titulo = pygame.font.SysFont("consolas", 36, bold=True)
    fuente_texto = pygame.font.SysFont("consolas", 20)

    # 3. Máquina de estados directa a la evaluación de huellas
    estado = "ESPERANDO_HUELLA"
    color_actual = COLOR_FONDO_NORMAL
    texto_principal = "ESPERANDO HUELLA DACTILAR"
    texto_secundario = "Presione 1-3 (Autorizado) o 4-6 (Intruso)"
    
    reloj = pygame.time.Clock()

    # 4. Bucle principal de eventos
    while True:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if evento.type == pygame.KEYDOWN:
                # --- ESTADO 1: EVALUAR HUELLA ---
                if estado == "ESPERANDO_HUELLA":
                    ruta_huella = obtener_ruta_prueba(evento.key, muestras_por_identidad)
                    
                    if ruta_huella:
                        if not ruta_huella.is_file():
                            texto_principal = "ERROR: ARCHIVO NO ENCONTRADO"
                            texto_secundario = f"Ruta faltante: {ruta_huella}"
                            color_actual = COLOR_ERROR
                        else:
                            # Petición de validación matemática
                            acceso, identidad = agente.evaluar_acceso(str(ruta_huella))
                            estado = "RESULTADO"
                            if acceso:
                                texto_principal = f"ACCESO CONCEDIDO: {identidad}"
                                color_actual = COLOR_EXITO
                            else:
                                texto_principal = "ACCESO DENEGADO (Intruso)"
                                color_actual = COLOR_ERROR
                            texto_secundario = "Presione [ENTER] para probar otra huella"

                # --- ESTADO 2: REINICIAR PRUEBA ---
                elif estado == "RESULTADO" and evento.key == pygame.K_RETURN:
                    estado = "ESPERANDO_HUELLA"
                    color_actual = COLOR_FONDO_NORMAL
                    texto_principal = "ESPERANDO HUELLA DACTILAR"
                    texto_secundario = "Presione 1, 2, 3 (Autorizado) o 4 (Intruso)"

        # --- DIBUJADO DE LA PANTALLA ---
        pantalla.fill(color_actual)
        superficie_prin = fuente_titulo.render(texto_principal, True, COLOR_TEXTO)
        superficie_sec = fuente_texto.render(texto_secundario, True, COLOR_TEXTO)
        
        pantalla.blit(superficie_prin, (ANCHO//2 - superficie_prin.get_width()//2, ALTO//2 - 40))
        pantalla.blit(superficie_sec, (ANCHO//2 - superficie_sec.get_width()//2, ALTO//2 + 30))
        pygame.display.flip()
        
        reloj.tick(30)

if __name__ == "__main__":
    main()