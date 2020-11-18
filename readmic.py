#!/usr/local/bin/python3
#pip install sounddevice
#pip install scipy
#Doc api sounddevice https://python-sounddevice.readthedocs.io/en/0.3.12/api.html

import sounddevice
from scipy.io.wavfile import write

def recsample(id=0):
    #frequence échantillonnage
    fs = 44100
    #durée en secondes
    sec = 4
    #Recupere les infos sur le micro integré dans un dictionnaire chans
    chans = sounddevice.query_devices(0,'input')
    print ("Enregistrement 4 secondes a 44100")
    print("Nombre channels sur divice :", chans)
    record_voice=sounddevice.rec(int(sec*fs),samplerate=fs,channels=chans["max_input_channels"])
    sounddevice.wait()
    s = "out" + str(id) + ".wav"
    write(s,fs,record_voice)
recsample()
