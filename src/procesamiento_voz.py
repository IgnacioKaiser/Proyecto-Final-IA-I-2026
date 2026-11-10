import os
import numpy as np
import librosa
import sounddevice as sd

# Configuración estándar para el audio
FRECUENCIA_MUESTREO = 44100
DURACION_SEGUNDOS = 3
N_MFCC = 13  # Cantidad de coeficientes Mel-Frequency Cepstral Coefficients
UMBRAL_RMS_SILENCIO = 0.003

def extraer_caracteristicas_archivo(ruta_archivo):
    """
    Carga un archivo .wav y extrae estadísticas temporales de sus MFCC.
    Retorna None si la muestra es silencio o demasiado corta.
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
    Extrae estadísticas de MFCC y sus deltas de una señal de audio.
    Devuelve None si la señal está en silencio o no tiene duración suficiente.
    """
    if senal.ndim > 1:
        senal = np.mean(senal, axis=1)
    senal = senal.astype(np.float32, copy=False)

    if sr <= 0 or senal.size == 0 or not np.all(np.isfinite(senal)):
        return None

    rms = float(np.sqrt(np.mean(np.square(senal, dtype=np.float64))))
    if rms < UMBRAL_RMS_SILENCIO:
        return None

    senal_sin_silencios, _ = librosa.effects.trim(senal, top_db=30)
    if senal_sin_silencios.size < int(sr * 0.2):
        return None

    mfccs = librosa.feature.mfcc(
        y=senal_sin_silencios,
        sr=sr,
        n_mfcc=N_MFCC,
    )
    ancho_delta = min(9, mfccs.shape[1] if mfccs.shape[1] % 2 else mfccs.shape[1] - 1)
    if ancho_delta >= 3:
        deltas = librosa.feature.delta(mfccs, width=ancho_delta)
    else:
        deltas = np.zeros_like(mfccs)

    return np.concatenate((
        np.mean(mfccs, axis=1),
        np.std(mfccs, axis=1),
        np.mean(deltas, axis=1),
        np.std(deltas, axis=1),
    ))

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
            archivos = sorted(f for f in os.listdir(ruta_comando) if f.lower().endswith(".wav"))
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
    Abre el micrófono durante el mismo intervalo que las muestras de entrenamiento
    y devuelve sus características acústicas, o None cuando detecta silencio.
    Se utilizará durante la ejecución principal del agente[cite: 4].
    """
    print("\n🎤 Escuchando comando...")
    grabacion = sd.rec(int(DURACION_SEGUNDOS * FRECUENCIA_MUESTREO),
                       samplerate=FRECUENCIA_MUESTREO, 
                       channels=1, 
                       dtype='float32')
    sd.wait() # Congela la ejecución hasta que terminen los 2 segundos
    print("✅ Procesando audio...")
    
    # Extraemos características de la señal grabada en RAM, sin guardarla en disco
    caracteristicas = extraer_caracteristicas_senal(grabacion, FRECUENCIA_MUESTREO)
    return caracteristicas