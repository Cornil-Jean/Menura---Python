import os
import pickle
from datetime import datetime, timedelta

historique_file_name = 'historique_file.pkl'
current_historique = []

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
Objet de formatage d'oiseau dans l'historique
"""
class bird_historique:
    def __init__(self, name, date):
        self.name = name
        self.date = date

"""
Fonction de récupération de l'historique
"""
def load_historique():
    if os.path.exists(historique_file_name):
        storage_file = open(historique_file_name, 'rb')
        current_historique = pickle.load(storage_file)
        storage_file.close()

        print("Hisotirque de détection :")
        if len(current_historique) > 0:
            for hist in current_historique:
                print(f"Oiseau : {hist.name} | date : {hist.date}")
        else:
            print("Pas d'historique")
        print("\n")

"""
Fonction de sauvgarde de l'historique
"""
def save_historique():
    if os.path.exists(historique_file_name):
        os.remove(historique_file_name)
    storage_file = open(historique_file_name, 'wb')
    pickle.dump(current_historique, storage_file)
    storage_file.close()

"""
Fonction de vérification d'oiseau

Si un oiseau est détecté il y a moin de 24h @return False
Si non @return True
"""
def check_historique(bird):
    if len(current_historique) > 0:
        for hist in current_historique:
            if (hist.name == bird.name):
                if bird.date <= (hist.date - timedelta(hours=24)):
                    hist.date = bird.date
                    return True
                else:
                    return False
        else:
            current_historique.append(bird)
            return True
    else:
        current_historique.append(bird)
        return True
