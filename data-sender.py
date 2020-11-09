import requests
import json
import os
from json import JSONEncoder
from datetime import date
import pickle
from uuid import getnode as get_mac

url_test = "http://www.google.com"
url_db = 'http://146.59.195.248:3000/v1/api/historiques'
timeout = 5
storage_file_name = 'data_storage_no_connection.p'

#
# Objet Oiseau pour sous format de la base de donnée
#
class Oiseau:
  def __init__(self, id, date, localisation, capteur):
      self.id = id
      self.date = date
      self.localisation = localisation
      self.capteur = capteur

#
# fonction de test de connection internet par le serveur google.com
#
def test_wifi_connection():
    try:
        requests.get(url_test, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout):
        return False

#
# envoies des data vers la base de donnée par requête post
# @input list of birds to send
#
def send_data_db(oiseau_list):

    print(oiseau_list)

    for oiseau_element in oiseau_list :
        print(oiseau_element)

        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        response = requests.post(url=url_db, data=oiseau_element, headers=headers)

        print("Status code: ", response.status_code)
        print(response.json())
        if (response.status_code != 200):
            return False
    os.remove(storage_file_name)
    return True

#
# Store data in file for future trensmit
# @input list of birds to store localy
#
def store_data(oiseau_list):
    storage_file = open(storage_file_name, 'wb')
    pickle.dump(oiseau_list, storage_file)
    storage_file.close()

#
# Load data from file
# @return the data if there is a file
# @return None in other cases
#
def recup_data():
    if os.path.exists(storage_file_name):
        storage_file = open(storage_file_name, 'rb')
        oiseau_loaded = pickle.load(storage_file)
        storage_file.close()

        print("Data from file :")
        print(oiseau_loaded)
        return oiseau_loaded
    return None

#
# fonction de dispatching en fonctione de l'état de la connection
# @input send data bird and load data if needed
#
def sendData(oiseau):
    oiseau_json = json.dumps({
        'oiseau': oiseau.id,
        'date': oiseau.date,
        'localisation': oiseau.localisation,
        'capteur':  oiseau.capteur}
    )

    oiseau_list_recup = recup_data()

    if oiseau_list_recup is not None:
        oiseau_list = oiseau_list_recup + [oiseau_json]
    else:
        oiseau_list = [oiseau_json]

    if test_wifi_connection():
        print("Connected to the Internet")
        print("sending data through the Internet")
        if(send_data_db(oiseau_list)):
            print("Envois des données éffectué")
        else:
            print("Erreur lors de l'envoie des données")
            store_data(oiseau_list)
    else:
        print("Erreur de connection au réseau")
        store_data(oiseau_list)

#
# main
#
def main():
    current_date = date.today()
    current_date_sqlFormat = current_date.strftime("%Y-%m-%d %H:%M:%S")
    oiseau = Oiseau(1, current_date_sqlFormat, 'testing' , "FF:FF:FF:FF:FF:FF") #get_mac()

    sendData(oiseau)

if __name__ == '__main__':
    main()