import requests
import json
from json import JSONEncoder
from datetime import date
from getmac import get_mac_address as gma

url_test = "http://www.google.com"
url_db = 'http://146.59.195.248:3000/v1/api/historiques'
timeout = 5

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
# Oiseau helper pour l'encodage en json
#
class OiseauEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__

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
#
#
def send_data_db(oiseau):

    oiseau_json = json.dumps({
        'oiseau': oiseau.id,
        'date': oiseau.date,
        'localisation': oiseau.localisation,
        'capteur': oiseau.capteur}
    )

    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response =requests.post(url=url_db, data=oiseau_json, headers=headers)

    print("Status code: ", response.status_code)
    print(response.json())
    if (response.status_code == 200):
        return True
    return False

#
# fonction de dispatching en fonctione de l'état de la connection
#
def sendData(oiseau):
    if test_wifi_connection():
        print("Connected to the Internet")
        print("sending data through the Internet")
        if(send_data_db(oiseau)):
            print("Envois des données éffectué")
        else:
            print("Erreur lors de l'envoie des données")

    else:
        print("No internet connection.")
        print("Storing data for future export")


#
# main
#
def main():
    current_date = date.today()
    current_date_sqlFormat = current_date.strftime("%Y-%m-%d %H:%M:%S")
    oiseau = Oiseau(1, current_date_sqlFormat, 'testing' , 1)

    sendData(oiseau)

if __name__ == '__main__':
    main()