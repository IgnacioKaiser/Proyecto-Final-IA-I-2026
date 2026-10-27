import pygame
import sys
import os
from src.agente import AgenteControlAcceso

# ==========================================
# CONFIGURACIÓN DE INTERFAZ (PYGAME)
# ==========================================
ANCHO, ALTO = 800, 600
COLOR_FONDO_NORMAL = (30, 30, 30)       # Gris oscuro (Dark mode)
COLOR_FONDO_ESCUCHA = (0, 100, 200)     # Azul
COLOR_EXITO = (0, 180, 50)              # Verde
COLOR_ERROR = (200, 0, 0)               # Rojo
COLOR_TEXTO = (240, 240, 240)           # Blanco suave

def obtener_ruta_prueba(tecla):
    """
    Simula el escáner físico. Devuelve la ruta de una huella de prueba
    dependiendo del número presionado en el teclado.
    """
    if tecla == pygame.K_1:
        return "data/huellas/prueba/autorizadas/autorizada_1_var_1.png"
    elif tecla == pygame.K_2:
        return "data/huellas/prueba/autorizadas/autorizada_2_var_1.png"
    elif tecla == pygame.K_3:
        return "data/huellas/prueba/autorizadas/autorizada_3_var_1.png"
    elif tecla == pygame.K_4:
        return "data/huellas/intrusos/id_desconocida_1/intrusa_1_var_1.png"
    return None

def main():
    print("=== INICIANDO SISTEMA MULTIMODAL ===")
    
    # 1. Instanciar y entrenar al Agente de IA
    # El umbral puede requerir ajuste dependiendo de los valores matemáticos de los Momentos de Hu.
    agente = AgenteControlAcceso(umbral_aceptacion=2.5) 
    agente.entrenar_kmeans()
    agente.entrenar_knn()

    # 2. Inicializar el motor gráfico
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pygame.display.set_caption("Agente de Control de Acceso Multimodal")
    
    # Cargar fuentes (usamos fuentes genéricas del sistema)
    fuente_titulo = pygame.font.SysFont("consolas", 36, bold=True)
    fuente_texto = pygame.font.SysFont("consolas", 20)

    # 3. Máquina de estados de la interfaz
    estado = "ESPERANDO_VOZ"
    color_actual = COLOR_FONDO_NORMAL
    texto_principal = "SISTEMA BLOQUEADO"
    texto_secundario = "Presione [ESPACIO] para activar el micrófono"
    
    reloj = pygame.time.Clock()

    # 4. Bucle principal de eventos
    while True:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
            if evento.type == pygame.KEYDOWN:
                # --- ESTADO 1: INICIAR ESCUCHA ---
                if estado == "ESPERANDO_VOZ" and evento.key == pygame.K_SPACE:
                    estado = "ESCUCHANDO"
                    color_actual = COLOR_FONDO_ESCUCHA
                    texto_principal = "ESCUCHANDO..."
                    texto_secundario = "Diga: 'reconocer', 'identificar' o 'salir'"
                    
                # --- ESTADO 2: SELECCIONAR HUELLA ---
                elif estado == "ESPERANDO_HUELLA":
                    ruta_huella = obtener_ruta_prueba(evento.key)
                    
                    if ruta_huella:
                        if not os.path.exists(ruta_huella):
                            texto_principal = "ERROR: ARCHIVO NO ENCONTRADO"
                            texto_secundario = f"No existe {ruta_huella}. ¿Ejecutó generador_sintetico.py?"
                            color_actual = COLOR_ERROR
                        else:
                            # Petición matemática de validación al agente
                            acceso, identidad = agente.evaluar_acceso(ruta_huella)
                            estado = "RESULTADO"
                            if acceso:
                                texto_principal = f"ACCESO CONCEDIDO: {identidad}"
                                color_actual = COLOR_EXITO
                            else:
                                texto_principal = "ACCESO DENEGADO (Intruso)"
                                color_actual = COLOR_ERROR
                            texto_secundario = "Presione [ENTER] para reiniciar"
                    else:
                        texto_secundario = "Presione 1, 2, 3 (Autorizados) o 4 (Intruso)"

                # --- ESTADO 3: REINICIAR ---
                elif estado == "RESULTADO" and evento.key == pygame.K_RETURN:
                    estado = "ESPERANDO_VOZ"
                    color_actual = COLOR_FONDO_NORMAL
                    texto_principal = "SISTEMA BLOQUEADO"
                    texto_secundario = "Presione [ESPACIO] para activar el micrófono"

        # --- DIBUJADO DE LA PANTALLA ---
        pantalla.fill(color_actual)
        superficie_prin = fuente_titulo.render(texto_principal, True, COLOR_TEXTO)
        superficie_sec = fuente_texto.render(texto_secundario, True, COLOR_TEXTO)
        
        pantalla.blit(superficie_prin, (ANCHO//2 - superficie_prin.get_width()//2, ALTO//2 - 40))
        pantalla.blit(superficie_sec, (ANCHO//2 - superficie_sec.get_width()//2, ALTO//2 + 30))
        pygame.display.flip()

        # --- LÓGICA DE AUDIO (Bloqueante) ---
        # Se ejecuta DESPUÉS de dibujar la pantalla azul para que el usuario sepa que puede hablar
        if estado == "ESCUCHANDO":
            comando = agente.escuchar_comando()
            
            if comando in ["reconocer", "identificar"]:
                estado = "ESPERANDO_HUELLA"
                color_actual = COLOR_FONDO_NORMAL
                texto_principal = f"COMANDO '{comando.upper()}' ACEPTADO"
                texto_secundario = "Simule escáner: Presione 1, 2, 3 (Autorizado) o 4 (Intruso)"
            elif comando == "salir":
                print("Cerrando sistema por comando de voz...")
                pygame.quit()
                sys.exit()
            else:
                estado = "ESPERANDO_VOZ"
                color_actual = COLOR_ERROR
                texto_principal = "COMANDO NO RECONOCIDO"
                texto_secundario = "Presione [ESPACIO] para intentar de nuevo"
                
        reloj.tick(30)

if __name__ == "__main__":
    main()