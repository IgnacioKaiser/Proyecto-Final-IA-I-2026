import sounddevice as sd
from scipy.io.wavfile import write
import os
import time

# Configuración
FRECUENCIA_MUESTREO = 44100
DURACION_SEGUNDOS = 3
COMANDOS = ["reconocer", "identificar", "salir"]
MUESTRAS_POR_COMANDO = 15 # Cantidad de muestras nuevas que agregarás cada vez que corras el script

def grabar_comandos():
    print("=== RECOLECTOR DE VOCES PARA K-NN ===")
    
    for comando in COMANDOS:
        carpeta_destino = os.path.join("data", "audio", comando)
        os.makedirs(carpeta_destino, exist_ok=True)
        
        # Cuenta cuántos archivos .wav ya existen en la carpeta
        archivos_existentes = len([f for f in os.listdir(carpeta_destino) if f.endswith('.wav')])
        inicio_indice = archivos_existentes + 1
        
        print(f"\nPrepárate para grabar el comando: '{comando.upper()}'")
        print(f"Ya tienes {archivos_existentes} muestras guardadas. Se agregarán {MUESTRAS_POR_COMANDO} nuevas.")
        time.sleep(2)
        
        for i in range(inicio_indice, inicio_indice + MUESTRAS_POR_COMANDO):
            # Ahora el contador de la interfaz refleja el número real del archivo
            input(f"Presiona Enter para grabar la muestra {i}...")
            print("🔴 Grabando... (Habla ahora)")
            
            # Graba el audio desde el micrófono
            grabacion = sd.rec(int(DURACION_SEGUNDOS * FRECUENCIA_MUESTREO), 
                               samplerate=FRECUENCIA_MUESTREO, 
                               channels=1, 
                               dtype='int16')
            sd.wait() # Espera a que termine la grabación
            print("✅ Grabación finalizada.")
            
            # Guarda el archivo .wav continuando la numeración
            ruta_archivo = os.path.join(carpeta_destino, f"muestra_{i}.wav")
            write(ruta_archivo, FRECUENCIA_MUESTREO, grabacion)

    print("\n¡Recolección completada! Revisa tu carpeta data/audio/")

if __name__ == "__main__":
    grabar_comandos()