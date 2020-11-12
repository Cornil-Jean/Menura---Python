#!/usr/local/bin/python3


#------------------------------#
# Liste des install a réaliser
#------------------------------#
#pip install sounddevice
#pip install scipy
#pip install opencv-python


#------------------------------#
# Documentation
#------------------------------#
#Doc api sounddevice https://python-sounddevice.readthedocs.io/en/0.3.12/api.html
#Doc OpenCV https://docs.opencv.org/master/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038

#------------------------------#
# Imports
#------------------------------#
import sounddevice
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
import cv2 as cv
import glob
import re

"""
Class de couleur pour l'écriture en console
"""
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

"""
Enregistement du sample

@output record_voice : array de l'audio enregistrée par le micro
@output fs : fréquance d'échantiollonage du sample retrounée
"""
def recsample():
    #frequence échantillonnage
    fs = 48000 / 1
    #durée en secondes
    sec = 4
    #Recupere les infos sur le micro integré dans un dictionnaire chans
    chans = sounddevice.query_devices(0,'input')
    print (f"Enregistrement {sec} secondes a {fs}dHz")
    record_voice=sounddevice.rec(int(sec*fs),samplerate=fs,channels=chans["max_input_channels"])
    # Attente de la fin du record du sample
    sounddevice.wait()
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

    # ajout de zéros au début des samples
    # basé sur la moitée de la taille de la fenêtre
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
    # cols pour la mise en place de la fenêtre
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # ajout de zéros a la fin des samples
    # basé sur la taille de la fenêtre
    samples = np.append(samples, np.zeros(frameSize))

    # mise en forme de la nouvelle array sur base des paramètres voulu par la FTT
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    # calcul de la FTT discrète sur base de la nouvelle array formatée
    return np.fft.rfft(frames)

"""
scale frequency axis logarithmically

@imput spec : spectrogram
@output newspac : new spectrogram
@output freqs :  frequence of the new spectrogram
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
def plotstft(sample, fs, binsize=2**10, colormap="Greys"):
    samplerate = fs
    samples = sample
    # signal par FTT
    s = stft(samples, binsize)

    # optention du nouveau signal sur base de la FTT
    # et de la fonction de représentation logarithmique du song
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    # trensformation de l'amplitude en décibel
    # pour la représentation sur graphique
    ims = 20.*np.log10(np.abs(sshow)/10e-6)

    # passage dans la fonction de netoyage de l'image
    ims = spectrogramCleaner(ims)

    # création du graphique avec plot
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # sauvgarde en pdf optionelle
    # plt.savefig("filename.pdf", bbox_inches = 'tight', pad_inches = 0)
    plt.savefig('ims.png', dpi= 400)
    plt.clf()

"""
Cleaner helper pour le mapping sur l'image

@ input value : valeur a traiter
@ input roof : valeur minimal de reset
@ output value traitée
"""
def cleanerSpectro_Helper(value, roof):
    if value < roof:
        return -roof
    return value ** 3


"""
Amélioration de l'image par convolution 2D

@input ims : array 2Darray représentant l'image du spectrogramme
@output ims : array traitée
"""
def spectrogramCleaner(ims):
    conv_array = np.array([
        [2, 1, 1],
        [1, 12, 1],
        [1, 1, 2]
    ])
    # passage par une array d'acentuation de l'image
    ims = signal.convolve2d(ims, conv_array, boundary='symm', mode='same')
    # clacul de la moyenne de l'image
    ims_mean = np.mean(ims)
    # suppression des éléments inférieurs a la moyenne
    # vectorization de la fonction pour sont exécution sur chaque cellule du tableau
    ims = np.vectorize(cleanerSpectro_Helper)(ims, ims_mean)
    # renvois de l'image nettoyée
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

        #définition de la méthode de matching
        # CCOEFF = coeficients de correlation
        # NORMED = au tour de 0 (pourcentage)
        meth = 'cv.TM_CCOEFF_NORMED'
        # application de la méthode matching
        method = eval(meth)

        # application de la correlation d'image
        res = cv.matchTemplate(ims,template,method)
        # récupération des valeurs de corrélation
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if max_val <  0.65 :
            print(f"with sample {sample} the coef of corr is : {bcolors.FAIL} {max_val}{bcolors.ENDC} \n")
        elif max_val >  0.65 and max_val <  0.90:
            print(f"with sample {sample} the coef of corr is : {bcolors.WARNING} {max_val}{bcolors.ENDC} \n")
        else:
            print(f"with sample {sample} the coef of corr is : {bcolors.OKGREEN} {max_val}{bcolors.ENDC} \n")
        # Si une meilleur valeure de corrélation est obtenure, on garde celle-ci
        if max_val > best_corr_val :
            best_corr_val = max_val
            best_corr_sample = sample

    # Vérification de la précision de la détection
    # si plus de X alors on considère que nous avons détecter quelque chose
    if best_corr_val < 0.30 :
        print("No correlation found")
        return None, None
    else :
        corr_acc = round(best_corr_val * 100, 3)
        print(f"\n|=================================================================================|"
              f"\n| Bird by correlation is : {bcolors.OKGREEN} {best_corr_sample} {bcolors.ENDC}"
              f"\n| with a Coef of Corr : {bcolors.WARNING} {corr_acc} {bcolors.ENDC}%"
              f"\n|=================================================================================|"
              f"\n"
              f"\n")
        # parsing de la correlation pour la récupération du nom de l'oiseau
        best_corr_sample = re.sub('_[0-9]*.png', '', best_corr_sample)
        best_corr_sample = best_corr_sample.replace('_', ' ')
        best_corr_sample = best_corr_sample.replace('samples-bank/', '')
        return best_corr_sample, corr_acc


"""
        Main
"""
sample, fs = recsample()
plotstft(sample, fs)

oiseau_name, corr_value = sampleCorrelation()
if( oiseau_name != None and corr_value != None) :
    print(f"Oiseau = {oiseau_name} | precision = {corr_value}")
