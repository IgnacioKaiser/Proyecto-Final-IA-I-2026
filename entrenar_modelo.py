# Contenido de entrenar_modelos.py
from src.agente import AgenteControlAcceso

def ejecutar_entrenamiento():
    print("=== INICIANDO FASE DE ENTRENAMIENTO (OFFLINE) ===")
    
    agente = AgenteControlAcceso()
    
    # 2. Procesar los datasets pesados
    agente.entrenar_kmeans()
    agente.entrenar_knn()
    metricas = agente.calibrar_umbral()
    print(f"\nResultados de validación: {metricas}")
    
    # 3. Guardar el estado en disco
    agente.guardar_modelos()
    
    print("\nEntrenamiento finalizado. Ahora puedes ejecutar main.py tranquilamente.")

if __name__ == "__main__":
    ejecutar_entrenamiento()