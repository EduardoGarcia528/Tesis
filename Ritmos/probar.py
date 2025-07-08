from music21 import converter
import numpy as np

filename = "data/humdrum-data/humdrum-data/beethoven/piano/sonata/beethoven-sonata01-1.krn"
score = converter.parse(filename)
