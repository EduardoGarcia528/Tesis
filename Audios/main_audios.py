import numpy as np
import unicodedata
import os
import re
import importlib
import funciones
importlib.reload(funciones)
from funciones import main, entropia_shannon
import xml.etree.ElementTree as ET
import subprocess
from music21 import stream, note, chord, meter, tempo, duration, instrument, clef, expressions, converter, environment
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as sg 
from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq, fftshift
import time

"""Funciones"""

def extraer_partitura_npy(carpeta):
    arrays = []

    # iteramos sobre los archivos
    for i, archivo in enumerate(os.listdir(carpeta)):
        ruta = os.path.join(carpeta, archivo)
        if os.path.isfile(ruta) and archivo.endswith(".npy"):
            # cargamos el array
            array = np.load(ruta)
            arrays.append(array)

    array_complete = np.array(arrays, dtype='object')

    return array_complete

def get_time_signature_from_offsets(arr):
    # Filtrar compás 1
    compas1 = arr[arr[:, 4] == 1]
    if compas1.size == 0:
        raise ValueError("No hay notas en el compás 1 para determinar el compás.")
    max_offset = np.max(compas1[:, 1])  # columna de offset
    numerador = int(round(max_offset))
    return f"{numerador}/4"

# Normalizador de texto: sin acentos, minúsculas, sin puntuación ni espacios extra
def normalize(text):
    text = text.lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')  # elimina acentos
    text = re.sub(r'[^a-z0-9]+', '', text)  # elimina puntuación y espacios
    return text
def get_tempo(term):
    norm = normalize(term)
    return tempo_dict.get(norm, None)

def midi_to_note(midi_val):
    return note.Note(midi_val) if midi_val >= 0 else note.Rest()

def array_to_voice(arr, time_signature):
    part = stream.Part()
    clef_to_use = choose_clef(arr[:, 3])
    part.append(clef_to_use)
    part.append(meter.TimeSignature(time_signature))

        
    current_measure_number = -1
    measure = None

    for measure_number in sorted(set(arr[:, -1])):
        measure = stream.Measure(number=int(measure_number))
        current_measure_number = measure_number

        # Extraer eventos de este compás
        notes_in_measure = arr[arr[:, -1] == measure_number]

        # Agrupar por onset dentro del compás
        onsets = sorted(set(notes_in_measure[:, 0]))
        for onset in onsets:
            group = notes_in_measure[notes_in_measure[:, 0] == onset]
            pitches = group[:, 3]
            durations = group[:, 2]

            if all(p == -1 for p in pitches):  # Todo es silencio
                r = note.Rest()
                r.duration = duration.Duration(durations[0])
                r.offset = onset
                measure.insert(onset, r)
            elif sum(p != -1 for p in pitches) == 1:
                idx = pitches != -1
                n = note.Note(int(pitches[idx][0]))
                n.duration = duration.Duration(durations[idx][0])
                n.offset = onset
                measure.insert(onset, n)
            else:
                chord_pitches = [int(p) for p in pitches if p != -1]
                dur_val = durations[pitches != -1][0]
                c = chord.Chord(chord_pitches)
                c.duration = duration.Duration(dur_val)
                c.offset = onset
                measure.insert(onset, c)

        part.append(measure)

    return part



def choose_clef(midi_vals):
    notes = [v for v in midi_vals if v >= 0]
    if not notes:
        return clef.TrebleClef()  # por defecto
    min_pitch = min(notes)
    max_pitch = max(notes)
    avg_pitch = sum(notes) / len(notes)

    # Algunas reglas simples (puedes ajustarlas a tus necesidades)
    if avg_pitch < 59:
        return clef.BassClef()
    elif avg_pitch > 62:
        return clef.TrebleClef()
    else:
        return clef.AltoClef()
    
def insertar_tempo_en_musicxml(path_in, path_out, tempo_bpm):
    tree = ET.parse(path_in)
    root = tree.getroot()

    # Namespaces de MusicXML
    ns = {'': 'http://www.musicxml.org/ns/musicxml'}

    # Buscar primer compás (measure) del primer part
    first_part = root.find('part')
    first_measure = first_part.find('measure')

    # Crear bloque <direction>
    direction = ET.Element('direction', attrib={'placement': 'above'})
    direction_type = ET.SubElement(direction, 'direction-type')
    metronome = ET.SubElement(direction_type, 'metronome')
    beat_unit = ET.SubElement(metronome, 'beat-unit')
    beat_unit.text = 'quarter'
    per_minute = ET.SubElement(metronome, 'per-minute')
    per_minute.text = str(tempo_bpm)

    sound = ET.SubElement(direction, 'sound', attrib={'tempo': str(tempo_bpm)})

    # Insertar al inicio del primer compás
    first_measure.insert(1, direction)

    # Guardar nuevo archivo
    tree.write(path_out, encoding='utf-8', xml_declaration=True)


def generador_partitura(array_3d,output_name,tempo,path):
    if path == False:
        score = stream.Score() 
        for i in range(np.shape(array_3d)[0]):
            arr = array_3d[i]
            print(arr)
            print(np.shape(arr))
            p = array_to_voice(arr, get_time_signature_from_offsets(arr))
            instrumento = "Piano"
            p.insert(0, instrument.fromString(instrumento))
            score.append(p)
        score.write('musicxml', fp='first_step.xml')
        insertar_tempo_en_musicxml('first_step.xml', output_name, tempo)
    else:
        score = stream.Score() 
        for filename in sorted(os.listdir(path)):
            if filename.endswith(".npy"):
                arr = np.load(f"{path}/{filename}")
                p = array_to_voice(arr, get_time_signature_from_offsets(arr))
                instrumento = f"{filename[:-11]}".replace("_", " ")
                # instrumento = "Piano"
                if instrumento == "Keyboard":
                    instrumento = "Piano"
                if instrumento == "StringInstrument":
                    instrumento = "Strings"
                if instrumento == "Brass":
                    instrumento = "Trumpet" 
                p.insert(0, instrument.fromString(instrumento))
                score.append(p)
        score.write('musicxml', fp='first_step.xml')
        insertar_tempo_en_musicxml('first_step.xml', output_name, tempo)

def mxl_to_wav(archivo_xml):
    # Establece la ruta al ejecutable de MuseScore 4
    environment.set('musescoreDirectPNGPath', r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")


    # Archivo de salida MP3
    archivo_wav = archivo_xml[:-3]+ "wav"

    # Llama a MuseScore desde la línea de comandos para exportar directamente a MP3
    subprocess.run([
        r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        archivo_xml,
        "-o",
        archivo_wav
    ])
    return archivo_wav

def generador_ruido_wav(duration,amplitude):
    # duration = 8          # Duración en segundos
    sample_rate = 44100   # Frecuencia de muestreo (Hz)
    # amplitude = 0.2       # Amplitud del ruido (entre 0 y 1)

    # Generar ruido blanco
    n_samples = duration * sample_rate
    noise = np.random.uniform(low=-1.0, high=1.0, size=n_samples)

    # Escalar a int16 para guardar como WAV
    noise_int16 = np.int16(noise * amplitude * 32767)

    # Guardar archivo
    write("ruido_blanco.wav", sample_rate, noise_int16)

    print("Archivo 'ruido_blanco.wav' generado correctamente.")
    return "ruido_blanco.wav"

def process_wav(archivo_wav, segundo2, minuto2 = 0.0):
    fs, señal = wav.read(archivo_wav) 
    if señal.ndim > 1:
        señal = np.mean(señal, axis=1)
    print(f'Frecuencia de muestreo: {fs} Hz')
    print(f'Duración: {len(señal) / fs} segundos')
    print(f'Forma de la señal: {señal.shape}')
    t = np.arange(len(señal)) / fs #eje x
    duracion = len(señal) / fs
    t_start= 0.0
    # minuto2, segundo2 = 0, duracion-3.0
    t_end = minuto2*60 + segundo2
    print('t_end',divmod(t_end, 60))
    ventana = (t >= t_start) & (t <= t_end) 
    N = int(fs*(t_end - t_start))
    muestra, t_muestra = sg.resample(señal[ventana], N, t[ventana]) 
    muestra = señal[:len(muestra)]
    print(len(muestra))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, señal)
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.title('Señal original')
    plt.axvspan(t_start, t_end, alpha=0.5, color='red') 

    plt.subplot(2, 1, 2)
    plt.plot(t_muestra, muestra)
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.title(f'Muestra de la señal S = {main(muestra,d=1,bivariante=False)}')

    plt.tight_layout()
    plt.show() 

"""""""""

"""""""""
ostinato = np.array([
        [0.0, 1.0, 1.0, 60.0, 1.0],
        [1.0, 2.0, 1.0, 64.0, 1.0],
        [2.0, 3.0, 1.0, 67.0, 1.0],
        [3.0, 4.0, 1.0, 64.0, 1.0],
        [0.0, 1.0, 1.0, 60.0, 2.0],
        [1.0, 2.0, 1.0, 64.0, 2.0],
        [2.0, 3.0, 1.0, 67.0, 2.0],
        [3.0, 4.0, 1.0, 64.0, 2.0],
        [0.0, 1.0, 1.0, 60.0, 3.0],
        [1.0, 2.0, 1.0, 64.0, 3.0],
        [2.0, 3.0, 1.0, 67.0, 3.0],
        [3.0, 4.0, 1.0, 64.0, 3.0],
        [0.0, 1.0, 1.0, 60.0, 4.0],
        [1.0, 2.0, 1.0, 64.0, 4.0],
        [2.0, 3.0, 1.0, 67.0, 4.0],
        [3.0, 4.0, 1.0, 64.0, 4.0]])

random_C_scale = np.array([
        [0.0, 1.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 1.0],
        [1.0, 2.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 1.0],
        [2.0, 3.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 1.0],
        [3.0, 4.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 1.0],
        [0.0, 1.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 2.0],
        [1.0, 2.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 2.0],
        [2.0, 3.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 2.0],
        [3.0, 4.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 2.0],
        [0.0, 1.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 3.0],
        [1.0, 2.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 3.0],
        [2.0, 3.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 3.0],
        [3.0, 4.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 3.0],
        [0.0, 1.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 4.0],
        [1.0, 2.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 4.0],
        [2.0, 3.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 4.0],
        [3.0, 4.0, 1.0, np.random.choice([60.0,62.,64.,65.,67.,69.,71.]), 4.0]])

random_12_note = np.array([
        [0.0, 1.0, 1.0, float(np.random.choice(np.arange(60,84))), 1.0],
        [1.0, 2.0, 1.0, float(np.random.choice(np.arange(60,84))), 1.0],
        [2.0, 3.0, 1.0, float(np.random.choice(np.arange(60,84))), 1.0],
        [3.0, 4.0, 1.0, float(np.random.choice(np.arange(60,84))), 1.0],
        [0.0, 1.0, 1.0, float(np.random.choice(np.arange(60,84))), 2.0],
        [1.0, 2.0, 1.0, float(np.random.choice(np.arange(60,84))), 2.0],
        [2.0, 3.0, 1.0, float(np.random.choice(np.arange(60,84))), 2.0],
        [3.0, 4.0, 1.0, float(np.random.choice(np.arange(60,84))), 2.0],
        [0.0, 1.0, 1.0, float(np.random.choice(np.arange(60,84))), 3.0],
        [1.0, 2.0, 1.0, float(np.random.choice(np.arange(60,84))), 3.0],
        [2.0, 3.0, 1.0, float(np.random.choice(np.arange(60,84))), 3.0],
        [3.0, 4.0, 1.0, float(np.random.choice(np.arange(60,84))), 3.0],
        [0.0, 1.0, 1.0, float(np.random.choice(np.arange(60,84))), 4.0],
        [1.0, 2.0, 1.0, float(np.random.choice(np.arange(60,84))), 4.0],
        [2.0, 3.0, 1.0, float(np.random.choice(np.arange(60,84))), 4.0],
        [3.0, 4.0, 1.0, float(np.random.choice(np.arange(60,84))), 4.0]])

scale_C_ascend = np.array([
        [0.0, 1.0, 1.0, 60.0, 1.0],
        [1.0, 2.0, 1.0, 62.0, 1.0],
        [2.0, 3.0, 1.0, 64.0, 1.0],
        [3.0, 4.0, 1.0, 65.0, 1.0],
        [0.0, 1.0, 1.0, 67.0, 2.0],
        [1.0, 2.0, 1.0, 69.0, 2.0],
        [2.0, 3.0, 1.0, 71.0, 2.0],
        [3.0, 4.0, 1.0, 72.0, 2.0],
        [0.0, 1.0, 1.0, 74.0, 3.0],
        [1.0, 2.0, 1.0, 76.0, 3.0],
        [2.0, 3.0, 1.0, 77.0, 3.0],
        [3.0, 4.0, 1.0, 79.0, 3.0],
        [0.0, 1.0, 1.0, 81.0, 4.0],
        [1.0, 2.0, 1.0, 83.0, 4.0],
        [2.0, 3.0, 1.0, 84.0, 4.0],
        [3.0, 4.0, 1.0, 86.0, 4.0]])

repetir = np.array([
        [0.0, 1.0, 1.0, 72.0, 1.0],
        [1.0, 2.0, 1.0, 60.0, 1.0],
        [2.0, 3.0, 1.0, 48.0, 1.0],
        [3.0, 4.0, 1.0, 84.0, 1.0],
        [0.0, 1.0, 1.0, 60.0, 2.0],
        [1.0, 2.0, 1.0, 60.0, 2.0],
        [2.0, 3.0, 1.0, 48.0, 2.0],
        [3.0, 4.0, 1.0, 60.0, 2.0],
        [0.0, 1.0, 1.0, 72.0, 3.0],
        [1.0, 2.0, 1.0, 72.0, 3.0],
        [2.0, 3.0, 1.0, 60.0, 3.0],
        [3.0, 4.0, 1.0, 48.0, 3.0],
        [0.0, 1.0, 1.0, 84.0, 4.0],
        [1.0, 2.0, 1.0, 72.0, 4.0],
        [2.0, 3.0, 1.0, 84.0, 4.0],
        [3.0, 4.0, 1.0, 60.0, 4.0],
])

acorde = np.array([
    [0.0,1.0,1.0,61.,1.0],
    [0.0,1.0,1.0,65.,1.0],
    [0.0,1.0,1.0,67.,1.0]])

    # [0.0,1.0,1.0,70.,1.0]])

array_complete = np.array((ostinato,))

partitura = r'data\humdrum-data-numpy\beethoven\piano\sonata\sonata14-3'
array_complete = extraer_partitura_npy(partitura)

mxl_file ='partitura.xml'
tempo = 120

generador_partitura(array_complete,output_name=mxl_file,tempo=tempo,path=False)

archivo_wav = mxl_to_wav(mxl_file)
# archivo_wav = generador_ruido_wav(duration=12,amplitude=0.3)
# process_wav(archivo_wav, segundo2=5.0) #15.5
process_wav(archivo_wav, segundo2=(60/tempo)*4*4) #15.5

for arch in [mxl_file,archivo_wav,'first_step.xml']:
    if os.path.exists(arch):
        os.remove(arch)
        print("Archivo eliminado.")
    else:
        print("El archivo no existe.")



