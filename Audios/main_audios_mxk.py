import numpy as np
import unicodedata
import os
import re
import importlib
import funciones
importlib.reload(funciones)
from funciones import entropia_shannon, extract_melody_grow, juntar_y_ordenar, extract_melody_simple
import funciones2
importlib.reload(funciones2)
from funciones2 import randomize_rhythm_per_bar_with_rests
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

from music21 import stream, meter, note, chord, duration
from fractions import Fraction
import numpy as np

def array_to_voice(arr):
    """
    arr: matriz Nx6 (dtype puede ser object)
      0: onset_local
      1: offset_local
      2: duration (quarterLength)
      3: midi
      4: measure
      5: timeSignature (string, ej. '4/4')
    """

    part = stream.Part()

    # Por si arr es dtype=object, casteamos columnas numéricas explícitamente
    onset_local  = arr[:, 0].astype(float)
    offset_local = arr[:, 1].astype(float)   # no la usamos aquí, pero la dejamos por claridad
    duration_col = arr[:, 2].astype(float)
    midi_col     = arr[:, 3].astype(float)
    measure_col  = arr[:, 4].astype(float)
    ts_col       = arr[:, 5]  # strings

    # Clef según los pitches de esta voz
    clef_to_use = choose_clef(midi_col)
    part.append(clef_to_use)

    # Lista ordenada de compases
    unique_measures = sorted(set(measure_col))

    last_ts_str = None  # para no repetir el mismo TS en cada compás

    for measure_number in unique_measures:
        # Crear Measure con número correcto
        m = stream.Measure(number=int(measure_number))

        # Filtrar eventos de este compás
        mask_m = (measure_col == measure_number)
        notes_in_measure = arr[mask_m]

        if notes_in_measure.shape[0] == 0:
            continue

        # Time signature de este compás (se asume consistente dentro del compás)
        ts_strings = np.unique(notes_in_measure[:, 5])
        ts_str = str(ts_strings[0])

        # Insertar TS si cambia (o si es el primero)
        if ts_str != last_ts_str:
            m.insert(0.0, meter.TimeSignature(ts_str))
            last_ts_str = ts_str

        # Recalcular subsets locales numéricos para este compás
        onsets_m    = notes_in_measure[:, 0].astype(float)  # onset_local
        durs_m      = notes_in_measure[:, 2].astype(float)  # duration (quarterLength)
        pitches_m   = notes_in_measure[:, 3].astype(float)  # midi

        # Agrupar por onset
        for onset in sorted(set(onsets_m)):
            mask_onset   = (onsets_m == onset)
            group_pitches = pitches_m[mask_onset]
            group_durs    = durs_m[mask_onset]

            # Caso: todos silencios
            if np.all(group_pitches == -1):
                r = note.Rest()
                r.quarterLength = group_durs[0]
                r.offset = onset
                m.insert(onset, r)

            # Caso: una única nota (no acorde)
            elif np.sum(group_pitches != -1) == 1:
                idx = np.where(group_pitches != -1)[0][0]
                n = note.Note(int(group_pitches[idx]))
                n.quarterLength = group_durs[idx]
                n.offset = onset
                m.insert(onset, n)

            # Caso: acorde
            else:
                chord_pitches = [int(p) for p in group_pitches if p != -1]
                dur_val = group_durs[group_pitches != -1][0]
                c = chord.Chord(chord_pitches)
                c.quarterLength = dur_val
                c.offset = onset
                m.insert(onset, c)

        part.append(m)

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

def generador_partitura(array_3d,tempo=120, output_name ='partitura.xml'):
    """
    array_3d: lista de matrices Nx6, una por voz.
      columnas:
        0: onset_local
        1: offset_local
        2: duration
        3: midi
        4: measure
        5: timeSignature (string, ej. '4/4', '3/8')
    """
    score = stream.Score()

    for i in range(len(array_3d)):
        arr = array_3d[i]
        # arr es Nx6 con ts en la última columna
        p = array_to_voice(arr)  # ya no pasamos time_signature global

        instrumento = "Piano"
        p.insert(0, instrument.fromString(instrumento))
        score.append(p)

    # Escribimos un archivo temporal y luego insertamos tempo
    score.write('musicxml', fp='first_step.xml')
    insertar_tempo_en_musicxml('first_step.xml', output_name, tempo)

def mxl_to_wav(archivo_xml = 'partitura.xml'):
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




from mxl_to_numpy import process_notes_by_voice # Nx6
from regla_simple_multivoz import extract_melody_multi_voices # Nx6

num = '1'
# partitura = r'data/moonlight_sonata_3rd_movement.mxl'
partitura = r'data/violin-concerto-in-d-minor-op-47-jean-sibelius-sibelius-violin-concerto-piano-and-violin.mxl'
partitura = r'data/fur-elise-violin.mxl'
partitura = f'data/new_scores/{num}.mxl'
score = converter.parse(partitura)

array_complete = process_notes_by_voice(score, start_measure=1) #mkl
print(len(array_complete))


# for i in range(np.shape(array_complete)[1]):
#     if array_complete[0][i,4] == 46.0:
#         print(array_complete[0][i,:])

melodia, serie1d = extract_melody_multi_voices(array_complete,del_repeats=False, del_loners=True)
# v = np.load('data\mel_brown.npy')[:450]
# print(len(serie1d))
# mask = melodia[:, 3] > 0        # True en las filas donde la col col_idx es positiva

# # Opcional: chequeo de seguridad
# n_positivos = mask.sum()
# if n_positivos != len(v):
#     raise ValueError(f"Número de positivos ({n_positivos}) != len(v) ({len(v)})")

# # Reemplazo
# melodia[mask, 3] = v

# mask = melodia[:, 4] == 22.0
# print(melodia[mask])

plt.plot(serie1d, marker = '.')
plt.xlabel('t')
plt.ylabel('MIDI')
plt.show()
# np.save(f'data/new_scores/melodies/{num}.npy',serie1d)
mxl_file = 'partitura.xml'

generador_partitura([melodia],tempo=120) # Nx6

archivo_wav = mxl_to_wav(mxl_file)

print(input("¿Eliminar?"))
for arch in [mxl_file,archivo_wav,'first_step.xml']:
    if os.path.exists(arch):
        os.remove(arch)
        print("Archivo eliminado.")
    else:
        print("El archivo no existe.")
