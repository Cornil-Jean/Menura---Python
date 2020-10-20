#!/usr/local/bin/python3
#pip install sounddevice
#pip install scipy
# pip install opencv-python
#Doc api sounddevice https://python-sounddevice.readthedocs.io/en/0.3.12/api.html
#Doc OpenCV https://docs.opencv.org/master/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038

import sounddevice
from scipy import signal
from scipy.io.wavfile import write
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
import cv2 as cv
import glob

"""
Enregistement du sample

@output record_voice : array de l'audio enregistrée par le micro 
@output fs : fréquance d'échantiollonage du sample retrounée
"""
def recsample():
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
    # s = "out" + str(id) + ".wav"
    # write(s,fs,record_voice)
    return record_voice, fs

"""
short time fourier transform of audio signal

@input sig : 
@input frameSize : 
@output : 
"""
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" 
scale frequency axis logarithmically

@imput spec : 
@output newspac : 
@output freqs :  
"""
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

"""
plot spectrogram

@input sample : array une dimension de l'audio a traiter
@input fs : fréquance d'échantillonage du sample
@output None (image sauvgardée sur le disque)
"""
def plotstft(sample, fs, binsize=2**10, plotpath=None, colormap="Greys"):
    samplerate = fs
    samples = sample

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    ims = spectrogramCleaner(ims)

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    plt.savefig('ims.png', dpi= 400)
    plt.clf()

"""
Cleaner helper

@ input value : valeur a traiter
@ input roof : valeur minimal de reset
@ output value traitée
"""
def cleanerHelper(value, roof):
    print(value)
    if value < roof:
        return 0
    return value ** 2

"""
Amélioration de l'image par convolution 2D

@input ims : array 2Darray représentant l'image du spectrogramme
@output ims : array traitée
"""
def spectrogramCleaner(ims):

    np.vectorize(cleanerHelper)(ims, 50)

    conv_array_push = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])

    ims = signal.convolve2d(ims, conv_array_push, boundary='symm', mode='same')

    np.vectorize(cleanerHelper)(ims, 250)

    return ims

"""
Correlation entre l'image donnée et la banque de donnée
"""
def sampleCorrelation():
    sampleList = glob.glob("samples-bank/*.png")

    # Récupération du spectrogramme a analyser
    ims_file = "ims.png"
    ims = cv.imread(ims_file,0)

    best_corr_val = 0
    best_corr_sample = ""

    for sample in sampleList :
        # import du sample
        template = cv.imread(sample,0)

        meth = 'cv.TM_CCOEFF_NORMED'
        method = eval(meth)

        # Apply template Matching
        res = cv.matchTemplate(ims,template,method)
        # Get Value from template matching
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        print(f"with sample {sample} the coef of corr is : {max_val} \n")

        # If better correlation save values
        if max_val > best_corr_val :
            best_corr_val = max_val
            best_corr_sample = sample

    # Test if corr coef is good inof
    if best_corr_val < 0.65 :
        print("No correlation found")
        return None, None
    else :
        print(f"\n |=================================================================================|"
              f"\n |Bird by correlation is : {best_corr_sample} "
              f"\n |with a Coef of Corr : {best_corr_val}|"
              f"\n |=================================================================================|")
        return best_corr_sample, best_corr_val


"""
        Main
"""

# récupération de l'audio
sample, fs = recsample()
plotstft(sample, fs)

corr_sample_name, corr_value = sampleCorrelation()
if( corr_sample_name != None and corr_value != None) :
    print(f"Sample = {corr_sample_name} | precision = {corr_value}")
else :
    print("Bird not found")