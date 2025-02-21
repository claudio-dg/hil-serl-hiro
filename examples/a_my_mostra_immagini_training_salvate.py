import pickle
import numpy as np
import matplotlib.pyplot as plt

# Path al file .pkl
file_path = "demo_data/pick_cube_sim_30_demos_2024-12-10_16-39-49.pkl"
# demo_data/pick_cube_sim_30_demos_2024-12-10_16-39-49.pkl
# classifier_data/pick_cube_sim_4_success_images_2024-12-11_09-35-44.pkl


# Funzione per caricare il file .pkl
def carica_pkl(file_path):
    with open(file_path, 'rb') as file:
        dati = pickle.load(file)
    return dati

# Caricamento dei dati
dati = carica_pkl(file_path)

# Esplora la struttura dei dati
print("Chiavi principali del dizionario:", dati[0].keys())
print("Chiavi di 'observations':", dati[0]['observations'].keys())

# Estrazione dell'immagine e stato
front_data = dati[0]['observations']['wrist']  # Batch di immagini SOSTITUIRE front / wrist per imagini esterne/dalPolso
state_data = dati[0]['observations']['state']  # Batch di stati

# Assicurati che esistano dati di immagini
if front_data is not None and len(front_data) > 0:
    print("Shape di 'front_data':", front_data.shape)
    # Visualizza il primo frame
    plt.imshow(front_data[0])  # Cambia indice per visualizzare altre immagini
    plt.title("Immagine 'front'")
    plt.colorbar()  # Aggiunge una barra colori
    plt.show()
else:
    print("Nessun dato di immagine trovato in 'front_data'.")
    


# Visualizzazione dei valori dello stato
if state_data is not None:
    print("Shape di 'state_data':", state_data.shape)
    print("Esempio di stato:", state_data[0])  # Cambia indice per vedere altri stati
else:
    print("Nessun dato di stato trovato in 'state_data'.")

