#!/usr/local/bin/python3


#------------------------------#
# Liste des install a réaliser
#------------------------------#
#pip install sounddevice
#pip install scipy
#pip install opencv-python
#pip install getmac

#------------------------------#
# Documentation
#------------------------------#
#Doc api sounddevice https://python-sounddevice.readthedocs.io/en/0.3.12/api.html
#Doc OpenCV https://docs.opencv.org/master/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038

#------------------------------#
# Imports
#------------------------------#
import math
import sounddevice
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
import cv2 as cv
import glob
import re
from datetime import date
from getmac import get_mac_address
import json
import dataSender

"""
Liste des oiseaux traité par l'api
"""
api_bird_list = [
    {"idoiseaux":1,"nom":"Mésange bleue"},
    {"idoiseaux":2,"nom":"Pic vert"},
    {"idoiseaux":3,"nom":"Moineau domestique"},
    {"idoiseaux":4,"nom":"Bergeronnette grise"},
    {"idoiseaux":5,"nom":"Buse variable"},
    {"idoiseaux":6,"nom":"Chardonneret élégant"},
    {"idoiseaux":7,"nom":"Bruant Jaune"},
]

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
    chans = sounddevice.query_devices(1,'input')
    print (f"Enregistrement {sec} secondes a {fs}dHz \n")
    record_voice=sounddevice.rec(int(sec*fs),samplerate=fs,channels=chans["max_input_channels"])
    # Attente de la fin du record du sample
    sounddevice.wait()
    return record_voice, fs

"""
short time fourier transform of audio signal

@input sig: signal su lequel exécuter la FTT
@input frameSize: taille du signal
@output : FTT de sig
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

@input spec: spectrogram
@output newspac: new spectrogram
@output freqs:  frequence of the new spectrogram
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

@input sample: array une dimension de l'audio a traiter
@input fs: fréquance d'échantillonage du sample
@output None: (image sauvgardée sur le disque)
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

    # calcul de la moyenne de l'image
    ims_mean = np.mean(ims)

    # suppression des éléments inférieurs a la moyenne
    # vectorization de la fonction pour sont exécution sur chaque cellule du tableau
    ims = np.vectorize(cleanerSpectro_Helper)(ims, ims_mean)


    # renvois de l'image nettoyée
    return ims

"""
Correlation entre l'image donnée et la banque de donnée

@input: None
@output: 
    best_corr_sample || None => Meilleur sample de correlation si existant
    corr_acc || None => Valeur de la correlation si existant
"""
def sampleCorrelation():
    # threshold d'acceptation de correlation
    # nombre max de correlatin acceptable
    corr_threshold = 0.6 # seul de correlation = 60%
    corr_value_bypass = 0.80  # seul de correlation assurée = 80%
    max_number_of_correlation = 5 # Max 5 correlation si non erreur
    # Récupération de la liste des samples de test
    sampleList = glob.glob("samples-bank/bird/*.png")
    # Récupération du sample de silence
    silence_sample = glob.glob("samples-bank/Silence.png")[0]

    # Récupération du spectrogramme a analyser
    ims_file = "ims.png"
    ims = cv.imread(ims_file,0)

    best_corr_val = 0
    best_corr_sample = ""

    # définition de la méthode de matching
    # CCOEFF = coeficients de correlation
    # NORMED = au tour de 0 (pourcentage)
    meth = 'cv.TM_CCOEFF_NORMED'
    # application de la méthode matching
    method = eval(meth)

    # Vérfication si détection d'oiseau
    # Vérification par comparaison avec le sample de silence
    # import du sample de silence comme template
    template = cv.imread(silence_sample,0)
    # application de la correlation d'image
    res = cv.matchTemplate(ims,template,method)
    # récupératin des valeurs de corrélation
    # min_val = pire coeff de correlation
    # max_val = meilleur coeff de correlation
    # min_loc = emplacement du pire coeff de correlation
    # max_loc = emplacement du meilleur coeff de correlation
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # Nous n'avons rien détecter
    if max_val > corr_threshold:
        return None, None
    else:
        number_of_correlation = 0
        best_corr_val = 0
        best_corr_delta = 0

        for sample in sampleList :
            # import du sample de test comme template
            template = cv.imread(sample,0)

            # application de la correlation d'image
            res = cv.matchTemplate(ims,template,method)
            # récupératin des valeurs de corrélation
            # min_val = pire coeff de correlation
            # max_val = meilleur coeff de correlation
            # min_loc = emplacement du pire coeff de correlation
            # max_loc = emplacement du meilleur coeff de correlation
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            # delta = différence maximum de correaltion
            delta = max_val - min_val

            if max_val < corr_threshold :
                print(f"with sample {sample} the coef of corr is : {bcolors.FAIL} {round(max_val,6)} {bcolors.ENDC} "
                      f"with delta: {round(delta,4)} \n")
            else:
                number_of_correlation += 1
                if delta > corr_threshold:
                    print(f"with sample {sample} the coef of corr is : {bcolors.OKGREEN} {round(max_val, 6)} {bcolors.ENDC} "
                          f"with delta: {bcolors.FAIL} {round(delta, 4)} {bcolors.ENDC} \n")
                else:
                    print(f"with sample {sample} the coef of corr is : {bcolors.OKGREEN} {round(max_val,6)} {bcolors.ENDC} "
                          f"with delta: {bcolors.OKGREEN} {round(delta,4)} {bcolors.ENDC} \n")
            # Si on a 90% de correlation avec le sample de test, on arret les corrspondances
            if (max_val > corr_value_bypass):
                best_corr_val = max_val
                best_corr_sample = sample
                number_of_correlation = -1
                break
            # Si une meilleur valeur de corrélation est obtenure
            # et que le delta de correlation est inférieur au threshold de correlation
            if max_val > best_corr_val and delta < corr_threshold:
                # si la valeur est plus grande de 10% de la valeur précédente de correlation
                # ou si le delta est inférieur au délta de correlation précédent
                if (math.floor(max_val*10) > math.floor(best_corr_val*10)) or (delta < best_corr_delta):
                    best_corr_val = max_val
                    best_corr_delta = delta
                    best_corr_sample = sample

        # Vérification de la précision de la détection
        # si plus que la valeur de corr_threshold alors on considère que nous avons détecter quelque chose
        # si plus de 3 valeurs étant considérés comme correlation => incertitude de la prédiction
        if best_corr_val < corr_threshold or number_of_correlation > max_number_of_correlation:
            print("No correlation found")
            return None, None

        # Si non nous avons détécé l'oiseau
        else :
            corr_acc = round(best_corr_val * 100, 3)
            print(f"\n|=================================================================================|"
                  f"\n| Bird by correlation is : {bcolors.OKGREEN} {best_corr_sample} {bcolors.ENDC}"
                  f"\n| with a Coef of Corr : {bcolors.WARNING} {corr_acc} {bcolors.ENDC}%"
                  f"\n|=================================================================================|"
                  f"\n"
                  f"\n")
            # parsing de la correlation pour la récupération du nom de l'oiseau
            best_corr_sample = best_corr_sample.replace('samples-bank/bird/', '')
            best_corr_sample = re.sub('_[0-9]*.png', '', best_corr_sample)
            best_corr_sample = best_corr_sample.replace('_', ' ')
            return best_corr_sample, corr_acc


"""
    Main
"""
# Record de l'audio
sample, fs = recsample()
plotstft(sample, fs)
# Correlation de l'audio
oiseau_name, corr_value = sampleCorrelation()
# Si nous avons détecter un oiseau
if( oiseau_name != None and corr_value != None) :
    print(f"Oiseau = {oiseau_name} | precision = {corr_value}")
    # récupération de la date de détection et parsing
    current_date = date.today()
    current_date_sqlFormat = current_date.strftime("%Y-%m-%d %H:%M:%S")
    # création d'un oiseau pour envois
    oiseau = dataSender.Oiseau(oiseau_name, current_date_sqlFormat, dataSender.get_location() , get_mac_address())
    print(json.dumps({
            'oiseau': oiseau.name,
            'date': oiseau.date,
            'localisation': oiseau.localisation,
            'capteur':  oiseau.capteur}
        ))
    # envois de l'oiseau a la db
    #dataSender.sendData(oiseau)
else:
    print("Aucun oiseau n'a été détecté")