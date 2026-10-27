# Contenido de entrenar_modelos.py
from src.agente import AgenteControlAcceso

def ejecutar_entrenamiento():
    print("=== INICIANDO FASE DE ENTRENAMIENTO (OFFLINE) ===")
    
    # 1. Instanciar el agente (puedes ajustar el umbral aquí si lo deseas)
    agente = AgenteControlAcceso(umbral_aceptacion=2.5)
    
    # 2. Procesar los datasets pesados
    agente.entrenar_kmeans()
    agente.entrenar_knn()
    
    # 3. Guardar el estado en disco
    agente.guardar_modelos()
    
    print("\nEntrenamiento finalizado. Ahora puedes ejecutar main.py tranquilamente.")

if __name__ == "__main__":
    ejecutar_entrenamiento()