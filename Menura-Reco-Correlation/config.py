import os
import pickle

config_file_name = 'config_file.pkl'
"""
value de l'entrée micro
"""
mic_entry_config = 0

"""
gestion des threads
"""
max_thread_config = 5
use_thread_config = True

"""
renvois nom de fichier de config
"""
def get_config_file_name():
    global config_file_name
    return config_file_name

"""
renvois de la configuration de l'application
"""
def get_config():
    load_config()
    return use_thread_config, max_thread_config, mic_entry_config

"""
setter mic
"""
def set_mic_entry(mic_entry):
    global mic_entry_config
    mic_entry_config = mic_entry

"""
setter max thread
"""
def set_max_thread(max_thread):
    global max_thread_config
    max_thread_config = max_thread

"""
setter use thread
"""
def set_use_thread(use_thread):
    global use_thread_config
    use_thread_config = use_thread

"""
sauvgadre de la configuratin dans un fichier
"""
def save_config(use_thread, max_thread, mic_entry):
    global mic_entry_config
    global max_thread_config
    global use_thread_config

    use_thread_config = use_thread
    max_thread_config = max_thread
    mic_entry_config = mic_entry

    #mise en forme pour sauvgarde
    settings = [use_thread_config, max_thread_config, mic_entry_config]

    if os.path.exists(config_file_name):
        os.remove(config_file_name)
    storage_file = open(config_file_name, 'wb')
    pickle.dump(settings, storage_file)
    storage_file.close()

"""
chargement de la configuration depuis le fichier de sauvgarde si existant
"""
def load_config():
    global mic_entry_config
    global max_thread_config
    global use_thread_config

    if os.path.exists(config_file_name):
        storage_file = open(config_file_name, 'rb')
        settings = pickle.load(storage_file)
        storage_file.close()
        # récupération des valeurs de config
        use_thread_config = settings[0]
        max_thread_config = settings[1]
        mic_entry_config = settings[2]
