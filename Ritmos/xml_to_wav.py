import subprocess
from music21 import converter, environment
import os

# Establece la ruta al ejecutable de MuseScore 4
environment.set('musescoreDirectPNGPath', r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")

# Ruta a tu archivo MusicXML
archivo_xml = "partitura_generada.xml"

# Archivo de salida MP3
archivo_wav = "partitura.wav"

# Llama a MuseScore desde la línea de comandos para exportar directamente a MP3
subprocess.run([
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    archivo_xml,
    "-o",
    archivo_wav
])
