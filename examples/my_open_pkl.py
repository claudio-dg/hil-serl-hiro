import pickle

import numpy as np
import matplotlib.pyplot as plt

# Percorso al file .pkl salvato
# file_path = "demo_data/pick_cube_sim_30_demos_2025-03-18_15-46-36.pkl"  # 69 elementi
# file_path = "demo_data/pick_cube_sim_30_demos_2025-03-26_15-01-00.pkl"  # 51 elementi


# file_path = "demo_data/A_my_UR_TEST_2_demos_first_completed.pkl" # 31316 elementi (?????)
# file_path = "demo_data/A_my_UR_TEST_2_demos_second_completed.pkl" # 8781 elementi (?????)

# ok che ci ho messo un minuto e mezzo a fare una sim perchè andava lenta, ma non son troppi?
#  anche perchè io ho fatto solo 2 Demo e non 20.. 
# secondo me qualche wrapper (che io non ho) cambia qualcosa


# file_path = "demo_data/A_my_UR_TEST_2_demos_2025-05-05_fixed_Actions.pkl" # 
# file_path = "demo_data/A_my_UR_TEST_2_demos_2025-05-05_fixedActions_e_consecutiveactFeature.pkl" # 
# file_path = "demo_data/A_my_UR_TEST_2_demos_2025-05-05_16-40-30.pkl" # 


################ classifier data from UR_rec_success_fail_sim.py ################

# ****** successi ******
# file_path = "classifier_data/pick_cube_sim_4_success_images_2024-12-11_09-35-44.pkl" 
# file_path = "classifier_data/A_my_UR_1_success_images_2025-05-06_14-45-52.pkl"  


# ****** failures ******
file_path = "classifier_data/A_my_UR_failure_images_2025-05-06_14-45-52.pkl" # 1678 elementi


# element_to_pick = 31300
element_to_pick = 1350

# Funzione per caricare il file .pkl
def carica_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def show_pkl_content(file_path):
    
    # Caricamento dei data
    data = carica_pkl(file_path)

    # Verifica il tipo e il contenuto
    print("Tipo del contenuto:", type(data))
    if isinstance(data, list):
        print(f"Il file contiene una lista con {len(data)} elementi. \n")
        # Analizza il primo elemento
        print("Tipo del elemento:", type(data[element_to_pick]) , " \n")
        print("Contenuto dell' elemento N:", element_to_pick, " = ", "\n\n ", data[element_to_pick])
        if isinstance(data[element_to_pick], dict):
            print("\n\n Chiavi del dizionario nel primo elemento:", data[element_to_pick].keys(), " \n")
            for key, value in data[element_to_pick].items():
                print(f"- {key}: Tipo = {type(value)}")
                if isinstance(value, np.ndarray):
                    print(f"       Forma = {value.shape}")
                    print(f"       Valori iniziali: {value.flatten()[:5]}")  # Mostra i primi 5 valori
    elif isinstance(data, dict):
        print("Chiavi del dizionario:", data.keys())
    else:
        print("Contenuto:", data)

def show_pkl_images(file_path):
    # Caricamento dei data
    data = carica_pkl(file_path)

    # Esplora la struttura dei data
    print("Chiavi principali del dizionario:", data[0].keys())
    print("Chiavi di 'observations':", data[0]['observations'].keys())

    # Estrazione dell'immagine e stato da aggiustare a mio caso sim, funziona con file vecchi x ora.. mio file salvato non ho salvato immagini!
    front_data = data[0]['observations']['wrist']  # Batch di immagini SOSTITUIRE front / wrist per imagini esterne/dalPolso
    state_data = data[0]['observations']['state']  # Batch di stati

    # Assicurati che esistano data di immagini
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



if __name__ == "__main__":
    show_pkl_content(file_path)
    # show_pkl_images(file_path)