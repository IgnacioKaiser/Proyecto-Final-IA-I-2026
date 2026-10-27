import os
import numpy as np
import librosa
import sounddevice as sd

# Configuración estándar para el audio
FRECUENCIA_MUESTREO = 44100
DURACION_SEGUNDOS_VIVO = 2
N_MFCC = 13  # Cantidad de coeficientes Mel-Frequency Cepstral Coefficients

def extraer_caracteristicas_archivo(ruta_archivo):
    """
    Carga un archivo de audio .wav desde el disco y extrae sus coeficientes MFCC.
    Retorna un vector unidimensional promediado en el tiempo.
    """
    try:
        # Cargar el audio. sr=None conserva la frecuencia de muestreo original del archivo
        senal, sr = librosa.load(ruta_archivo, sr=None)
        return extraer_caracteristicas_senal(senal, sr)
    except Exception as e:
        print(f"Error al procesar {ruta_archivo}: {e}")
        return None

def extraer_caracteristicas_senal(senal, sr):
    """
    Extrae los MFCC directamente de un arreglo (matriz) de audio[cite: 4].
    Esto es útil tanto para archivos cargados como para grabaciones en vivo.
    """
    # Si la señal de sounddevice es estéreo o tiene forma (N, 1), la aplanamos a 1D
    if senal.ndim > 1:
        senal = senal.flatten()
        
    # Extraer MFCCs usando librosa
    # librosa devuelve una matriz de dimensiones (n_mfcc, frames_de_tiempo)
    mfccs = librosa.feature.mfcc(y=senal.astype(np.float32), sr=sr, n_mfcc=N_MFCC)
    
    # Para que K-NN pueda procesarlo, necesitamos un único vector fijo por audio.
    # Promediamos los valores a lo largo del eje del tiempo (axis=1)
    mfccs_promedio = np.mean(mfccs, axis=1)
    
    return mfccs_promedio

def cargar_dataset_voz(directorio_base="data/audio"):
    """
    Recorre las carpetas ("reconocer", "identificar", "salir"), procesa cada .wav
    y devuelve las matrices X (características) e y (etiquetas) listas para K-NN[cite: 4].
    """
    X = []
    y = []
    
    if not os.path.exists(directorio_base):
        print(f"Advertencia: No se encontró la ruta '{directorio_base}'.")
        return np.array(X), np.array(y)

    comandos = os.listdir(directorio_base)
    
    for comando in comandos:
        ruta_comando = os.path.join(directorio_base, comando)
        
        # Ignorar si hay archivos sueltos que no sean carpetas
        if os.path.isdir(ruta_comando):
            archivos = [f for f in os.listdir(ruta_comando) if f.endswith(".wav")]
            print(f"Procesando {len(archivos)} muestras para el comando '{comando}'...")
            
            for archivo in archivos:
                ruta_archivo = os.path.join(ruta_comando, archivo)
                caracteristicas = extraer_caracteristicas_archivo(ruta_archivo)
                
                if caracteristicas is not None:
                    X.append(caracteristicas)
                    y.append(comando)  # La etiqueta es la palabra ("reconocer", "identificar" o "salir")
                    
    return np.array(X), np.array(y)

def escuchar_audio_en_vivo():
    """
    Abre el micrófono, graba durante 2 segundos y devuelve las características acústicas.
    Se utilizará durante la ejecución principal del agente[cite: 4].
    """
    print("\n🎤 Escuchando comando...")
    grabacion = sd.rec(int(DURACION_SEGUNDOS_VIVO * FRECUENCIA_MUESTREO), 
                       samplerate=FRECUENCIA_MUESTREO, 
                       channels=1, 
                       dtype='float32')
    sd.wait() # Congela la ejecución hasta que terminen los 2 segundos
    print("✅ Procesando audio...")
    
    # Extraemos características de la señal grabada en RAM, sin guardarla en disco
    caracteristicas = extraer_caracteristicas_senal(grabacion, FRECUENCIA_MUESTREO)
    return caracteristicas