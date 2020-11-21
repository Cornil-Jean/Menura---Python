import time
from getmac import get_mac_address
import sounddevice
from art import *

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
Helper pour le print de l'intro au lancement de l'app
"""
def intro_printer():
    tprint("Menura")
    tprint("       Bird-Tracker")
    print("")
    print(f"{bcolors.BOLD}{bcolors.HEADER}|===================================================================| {bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}|                                                                   | {bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}| Bienvenu dans notre application de reconnaissance des oiseaux     | {bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}|                                                                   | {bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}| N'oubliez pas notre application sur votre smartphone              | {bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}|                                                                   | {bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}|              Menura: Bird-Tracker                                 | {bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}|                            sur IOS et Android                     | {bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}|                                                                   | {bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.HEADER}|===================================================================| {bcolors.ENDC}")
    print("")
    print(f"{bcolors.BOLD}{bcolors.WARNING}|===================================================================| {bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.WARNING}| Appuyez sur Ctrl-C pour fermer l'application                      |{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.WARNING}|===================================================================| {bcolors.ENDC}")

"""
Affichage de l'historique pour le menu
"""
def show_historique(current_historique):
    tprint("Historique")
    print(f"{bcolors.OKCYAN}"
          f"Hisotirque de détection :"
          f"{bcolors.ENDC}")
    print(f"{bcolors.BOLD}{bcolors.WARNING}"
          f"--------------------------"
          f"{bcolors.ENDC}")
    if len(current_historique) > 0:
        for hist in current_historique:
            print(f"Oiseau : {hist.name} | date : {hist.date}")
    else:
        print("Pas d'historique actuellement enregistré sur cette appareil")
    input(f"\n \n{bcolors.WARNING}entrez une touche pour quitter {bcolors.ENDC}\n")

"""
Affichage et renvois des information sur les thread pour le menu
"""
def show_thread(use_thread, max_thread):
    tprint("Multit Thread")
    print(f"\n{bcolors.OKCYAN}{bcolors.BOLD}")
    print(f"Configuration actuelle")
    print(f"{bcolors.ENDC} {bcolors.OKCYAN}")
    print(f" Multit Thread :              {use_thread}")
    print(f" nombre de Threads :          {max_thread}")
    thread = input(f"\n \n{bcolors.WARNING}Entrez le nombre de thread a utiliser \n (pour ne pas utiliser le multit tread entrez 0) {bcolors.ENDC}\n")
    return int(thread)

"""
Affichage et renvois des information sur l'entrée micro pour le menu
"""
def show_mic_entry(mic_entry):
    tprint("Périphériques")
    print(f"\n{bcolors.OKCYAN}{bcolors.BOLD}")
    print(f" Configuration actuelle")
    print(f"{bcolors.ENDC} {bcolors.OKCYAN}")
    print(f" nom périphérique utilisé:    {sounddevice.query_devices(mic_entry)['name']}")
    print(f" numéro périphérique utilisé: {mic_entry}")

    print(f"{bcolors.OKCYAN} {bcolors.BOLD}")
    print(f" Liste des périphériques disponibles : \n")
    print(f" ID | Nom du périphérique {bcolors.ENDC}{bcolors.OKCYAN}")
    print(sounddevice.query_devices())
    print(f"{bcolors.ENDC}")
    choix_entry = input(f"\n \n{bcolors.WARNING}Entrer l'ID du périphérique a utiliser {bcolors.ENDC}\n")
    return int(choix_entry)

"""
Affichage de la mac adresse pour le menu et l'applicatin
"""
def show_mac_add():
    tprint("MAC adresse")
    mac_add = get_mac_address()
    print(f"{bcolors.FAIL}{bcolors.OKCYAN}Votre addresse mac est : {mac_add}{bcolors.ENDC}")
    print("\n")
    print("Afin de lier votre application et votre capteur, \n"
          f"veuillez entrer cette adresse dans la partie \"paramètre\" de votre application dans la secetion \"capteur\" ")
    input(f"\n \n{bcolors.WARNING}entrez une touche pour quitter {bcolors.ENDC}\n")

def config_printer(use_thread, max_thread, mic_entry):
    print(f"\n{bcolors.OKCYAN}{bcolors.BOLD}")
    print(f" Configuration actuelle")
    print(f"{bcolors.ENDC} {bcolors.OKCYAN}")
    print(f" Multit Thread :              {use_thread}")
    print(f" nombre de Threads :          {max_thread}")

    print(f" nom périphérique utilisé:    {sounddevice.query_devices(mic_entry)['name']}")
    print(f" numéro périphérique utilisé: {mic_entry}")
    print(f"{bcolors.ENDC}")

"""
Helper pour le choix de l'entrée du menu
"""
def choice_printer():
    print(f"{bcolors.BOLD}{bcolors.HEADER}")
    print(f"|================================|")
    print(f"|                                |")
    print(f"|    Que voulez-vous faire ?     |")
    print(f"|                                |")
    print(f"|================================|")
    print(f"|                                |")
    print(f"| 1) mon historique              |")
    print(f"| 2) multit thread               |")
    print(f"| 3) choix de l'entrée micro     |")
    print(f"| 4) mon adresse MAC             |")
    print(f"|                                |")
    print(f"| 0) Lancer la détection         |")
    print(f"|================================|")
    print(f"{bcolors.ENDC} \n \n", end='\r')


