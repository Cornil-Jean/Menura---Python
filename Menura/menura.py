#!/usr/local/bin/python3


#------------------------------#
# Liste des install a réaliser
#------------------------------#
#pip install sounddevice
#pip install scipy
#pip install opencv-python
#pip install getmac
#pip install art

#------------------------------#
# Documentation
#------------------------------#
#Doc api sounddevice https://python-sounddevice.readthedocs.io/en/0.3.12/api.html
#Doc OpenCV https://docs.opencv.org/master/df/dfb/group__imgproc__object.html#gga3a7850640f1fe1f58fe91a2d7583695dac6677e2af5e0fae82cc5339bfaef5038

#------------------------------#
# Imports
#------------------------------#
import sys
import os
import time
import threading
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
import historique
import menu_render
import config

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
value de l'entrée micro
"""
mic_entry = 0

"""
gestion des threads
"""
max_thread = 5
use_thread = True

"""
setter mic entry
"""
def set_mic_entry(mic_entry_value):
    global  mic_entry
    mic_entry = mic_entry_value

"""
setter max thread
"""
def set_max_thread(max_thread_value):
    global  max_thread
    max_thread = max_thread_value

"""
setter use thread
"""
def set_use_thread(use_thread_value):
    global  use_thread
    use_thread = use_thread_value

"""
Class de couleur pour l'écriture en console
"""
class bcolors:
    HEADER = '\033[95m'
    WARNING = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

"""
fonction helper clear console
"""
def cls():
    os.system('cls' if os.name=='nt' else 'clear')

"""
Enregistement du sample

@output record_voice : array de l'audio enregistrée par le micro
@output fs : fréquance d'échantiollonage du sample retrounée
"""
def recsample():
    #frequence échantillonnage
    fs = 48000
    #durée en secondes
    sec = 4
    #Recupere les infos sur le micro integré dans un dictionnaire chans
    chans = sounddevice.query_devices(mic_entry,'input')
    print (f"Enregistrement {sec} secondes a {fs}dHz \n")
    record_voice=sounddevice.rec(int(sec*fs),samplerate=fs,channels=chans["max_input_channels"])
    # Attente de la fin du record du sample
    sounddevice.wait()
    return record_voice, fs

"""
short time fourier transform of audio signal

passage de temporel a fréquance

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
    plt.savefig("ims.png", dpi= 400)
    plt.clf()
    plt.cla()
    plt.close('all')

    ims_plot_data = cv.imread('ims.png', 0)
    return ims_plot_data

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
Parsing du nom des oiseaux
"""
def pars_bird_name(name):
    name = name.replace('samples-bank/bird/', '')
    name = re.sub('_[0-9]*.png', '', name)
    name = name.replace('_', ' ')
    return name

"""
Correlation entre l'image donnée et la banque de donnée

@input: None
@output: 
    best_corr_sample || None => Meilleur sample de correlation si existant
    corr_acc || None => Valeur de la correlation si existant
"""
def sampleCorrelation(ims_plot_data, verbose):
    # threshold d'acceptation de correlation
    # nombre max de correlatin acceptable
    corr_threshold = 0.6 # seul de correlation = 60%
    corr_value_bypass = 0.80  # seul de correlation assurée = 80%
    max_number_of_correlation = 10 # Max 10 correlation si non erreur
    # liste de détection des oiseaux
    # et variables utilisées lors de la correlation
    current_detection = []
    best_corr_val = corr_threshold
    best_corr_delta = 0
    number_of_correlation = 0
    # Récupération de la liste des samples de test
    sampleList = glob.glob("samples-bank/bird/*.png")
    # Récupération du sample de silence
    silence_sample = glob.glob("samples-bank/Silence.png")[0]

    # Récupération du spectrogramme a analyser
    #ims_file = ims_file_name
    #ims = cv.imread(ims_file,0)
    ims = ims_plot_data
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

            if max_val < corr_threshold and verbose:
                print(f"with sample {sample} the coef of corr is : {bcolors.FAIL} {round(max_val,6)} {bcolors.ENDC} "
                      f"with delta: {round(delta,4)} \n")
            else:
                # détection rentre dans les cirtère de valeurs
                number_of_correlation += 1
                # si trop de détection possible, arrêt des test de correlation
                if number_of_correlation > max_number_of_correlation:
                    break
                if delta > corr_threshold and verbose:
                    print(f"with sample {sample} the coef of corr is : {bcolors.OKGREEN} {round(max_val, 6)} {bcolors.ENDC} "
                          f"with delta: {bcolors.FAIL} {round(delta, 4)} {bcolors.ENDC} \n")
                elif verbose:
                    print(f"with sample {sample} the coef of corr is : {bcolors.OKGREEN} {round(max_val,6)} {bcolors.ENDC} "
                          f"with delta: {bcolors.OKGREEN} {round(delta,4)} {bcolors.ENDC} \n")
            # Si on a 80% de correlation avec le sample de test, on arret les correspondances
            if (max_val > corr_value_bypass):
                best_corr_val = max_val
                best_corr_sample = sample
                break
            # Si une meilleur valeur de corrélation est obtenure
            if max_val > best_corr_val :
                # si l'oiseau a déjà été détecté on ajoute de 5% de probabilitées
                if (pars_bird_name(sample) in current_detection):
                    if verbose:
                        print(f"{bcolors.OKCYAN}{pars_bird_name(sample)} détecté, amélioration des probabilitées {bcolors.ENDC}")
                    max_val += 0.05
                    number_of_correlation -= 1
                else:
                    # ajout a la liste des oiseaux de correlation
                    current_detection.append(pars_bird_name(sample))
                # attribution comme meilleure correlation
                best_corr_val = max_val
                best_corr_delta = delta
                best_corr_sample = sample

        # Vérification de la précision de la détection
        # si plus que la valeur de corr_threshold alors on considère que nous avons détecter quelque chose
        # si plus de 3 valeurs étant considérés comme correlation => incertitude de la prédiction
        if best_corr_val < corr_threshold or number_of_correlation > max_number_of_correlation:
            return None, None

        # Si non nous avons détécé l'oiseau
        else :
            corr_acc = round(best_corr_val * 100, 3)
            # parsing de la correlation pour la récupération du nom de l'oiseau
            best_corr_sample = pars_bird_name(best_corr_sample)
            print(f"\n{bcolors.WARNING}|=================================================================================|"
                  f"\n{bcolors.WARNING}| Oiseau détecté par correlation : {bcolors.OKGREEN} {best_corr_sample} {bcolors.ENDC}"
                  f"\n{bcolors.WARNING}| Précision de la correlation : {bcolors.WARNING} {corr_acc} {bcolors.ENDC}%"
                  f"\n{bcolors.WARNING}| Delta de la correlation : {bcolors.WARNING} {round(best_corr_delta, 5)} {bcolors.ENDC}"
                  f"\n{bcolors.WARNING}|=================================================================================|{bcolors.ENDC}"
                  f"\n"
                  f"\n")
            return best_corr_sample, corr_acc

def correlation(ims_plot_data, verbose):
    # Correlation de l'audio
    oiseau_name, corr_value = sampleCorrelation(ims_plot_data, verbose)
    # Si nous avons détecter un oiseau
    if (oiseau_name != None and corr_value != None):
        # récupération de la date de détection et parsing
        current_date = date.today()
        # création d'un oiseau pour le check de l'historique
        oiseau_historique = historique.bird_historique(oiseau_name, current_date)
        if historique.check_historique(oiseau_historique):
            current_date_sqlFormat = current_date.strftime("%Y-%m-%d %H:%M:%S")
            # création d'un oiseau pour envois
            oiseau = dataSender.Oiseau(oiseau_name, current_date_sqlFormat, dataSender.get_location(), get_mac_address())
            if verbose:
                print(json.dumps({
                    'oiseau': oiseau.name,
                    'date': oiseau.date,
                    'localisation': oiseau.localisation,
                    'capteur': oiseau.capteur}
                ))
                print("\n")
            # envois de l'oiseau a la db
            dataSender.sendData(oiseau, verbose)
        else:
            print(f"\n {bcolors.OKCYAN} {oiseau_name} déjà détécté il y a moin de 24h {bcolors.ENDC} \n")
    else:
        print("Aucun oiseau n'a été détecté")

"""
Gestion de l'interface menu en mode verbose
"""
def choice_handler():
    global mic_entry
    global use_thread
    global max_thread

    config.save_config(use_thread, max_thread, mic_entry)

    menu_render.intro_printer()
    menu_render.config_printer(use_thread, max_thread, mic_entry)
    menu_render.choice_printer()
    choix = input(" ")

    if (choix == "1"):
        cls()
        menu_render.show_historique(historique.get_historique())
        cls()
        choice_handler()
    elif (choix == "2"):
        cls()
        thread = menu_render.show_thread(use_thread, max_thread)
        if (thread == 0):
            use_thread = False
            config.set_use_thread(False)
        else:
            use_thread = True
            config.set_use_thread(True)
        max_thread = thread
        config.set_max_thread(thread)
        cls()
        choice_handler()
    elif (choix == "3"):
        cls()
        mic_entry = menu_render.show_mic_entry(mic_entry)
        config.set_mic_entry(mic_entry)
        cls()
        choice_handler()
    elif (choix == "4"):
        cls()
        menu_render.show_mac_add()
        cls()
        choice_handler()
    elif (choix == "5"):
        cls()
        menu_render.show_bird_list()
        cls()
        choice_handler()

"""
    Main
    
    To execute in verbose mode add one of the current arguments :
                    -v -V --verbose --Verbose
    
"""
def main():
    verbose = False
    if (len(sys.argv) > 1):
        if (re.match(r"[-v,-V]",sys.argv[1])) or (re.match(r"--[v,V](erbose)",sys.argv[1])):
            print(f"\n\n{bcolors.OKGREEN}Executing in verbose mode{bcolors.ENDC}\n")
            verbose = True

    try:
        menu_render.intro_printer()
        # chargement de l'historique
        historique.load_historique()
        # récupération des configurations
        use_thread_value, max_thread_value, mic_entry_value = config.get_config()
        set_mic_entry(mic_entry_value)
        set_max_thread(max_thread_value)
        set_use_thread(use_thread_value)
        if verbose or not os.path.exists(config.get_config_file_name()):
            cls()
            choice_handler()
        if dataSender.test_wifi_connection():
            dataSender.save_location(None)
        cls()
        while True:
            # Record de l'audio
            sample, fs = recsample()
            ims_plot_data = plotstft(sample, fs)
            ims_file_name = 'ims.png'

            # threading de correlation
            if (threading.activeCount() < max_thread and use_thread):
                if verbose:
                    print(f'{bcolors.OKCYAN}Total of current tread : {threading.active_count()} of {max_thread} {bcolors.ENDC} \n')
                t = threading.Thread(target=correlation, args=(ims_plot_data, verbose,))
                t.start()
            else:
                correlation(ims_plot_data, verbose)


    except KeyboardInterrupt as e:
        print("\n")
        print(f"{bcolors.FAIL}{bcolors.BOLD}|=====================================================|{bcolors.ENDC}")
        print(f"{bcolors.FAIL}{bcolors.BOLD}|Veuillez attendre la fin de la fermeture du programme|{bcolors.ENDC}")
        print(f"{bcolors.FAIL}{bcolors.BOLD}|                                                     |{bcolors.ENDC}")
        print(f"{bcolors.FAIL}{bcolors.BOLD}|Les dernières comparaisons sont en cours de calcul   |{bcolors.ENDC}")
        print(f"{bcolors.FAIL}{bcolors.BOLD}|=====================================================|{bcolors.ENDC}")
        historique.save_historique()
        sys.exit(e)
        pass




if __name__ == '__main__':
    main()