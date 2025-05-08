import pickle

import numpy as np
import matplotlib.pyplot as plt

####################################################################################################
################################ demo data from UR_rec_demos_sim.py ################################

# file_path = "demo_data/pick_cube_sim_30_demos_2025-03-18_15-46-36.pkl"  # 69 elementi --> ma non è realmente 30 demo mi sa
# file_path = "demo_data/pick_cube_sim_30_demos_2025-03-26_15-01-00.pkl"  # 51 elementi
# file_path = "demo_data/pick_cube_sim_30_demos_2024-12-10_16-39-49.pkl"  #  1782 elementi -Z questo è 30 DEMOS !

# file_path = "demo_data/A_my_UR_TEST_2_demos_first_completed.pkl" # 31316 elementi (?????)
# file_path = "demo_data/A_my_UR_TEST_2_demos_second_completed.pkl" # 8781 elementi (?????)

# ok che ci ho messo un minuto e mezzo a fare una sim perchè andava lenta, ma non son troppi?
#  anche perchè io ho fatto solo 2 Demo e non 20.. 
# secondo me qualche wrapper (che io non ho) cambia qualcosa


# file_path = "demo_data/A_my_UR_TEST_2_demos_2025-05-05_fixed_Actions.pkl" # 
# file_path = "demo_data/A_my_UR_TEST_2_demos_2025-05-05_fixedActions_e_consecutiveactFeature.pkl" # 
# file_path = "demo_data/A_my_UR_TEST_2_demos_2025-05-05_16-40-30.pkl" # 

file_path = "demo_data/A_my_UR_TEST_2_demos_1ogni100_2025-05-08_13-57-08.pkl" # (HA IMMAGINI) 129 elementi per 2 tentativi

#################################################################################################################
################################ classifier data from UR_rec_success_fail_sim.py ################################

# ****** successi ******
# file_path = "classifier_data/pick_cube_sim_4_success_images_2024-12-11_09-35-44.pkl" 
# file_path = "classifier_data/A_my_UR_1_success_images_2025-05-06_14-45-52.pkl"  
# file_path = "classifier_data/A_my_UR_30_success_Con_IMMAGINI_2025-05-07_11-00-17.pkl"  

# ****** failures ******
# file_path = "classifier_data/A_my_UR_failure_images_2025-05-06_14-45-52.pkl" # 1678 elementi
# file_path = "classifier_data/A_my_UR_failure_images_1ogni100failures.pkl" # 10 elementi
# file_path = "classifier_data/A_my_UR_failure_Con_IMMAGINI_2025-05-07_11-00-17.pkl" # 28 elementi
file_path = "classifier_data/A_my_UR_failure_images_2025-05-08_17-05-05.pkl" # 10 elementi
# element_to_pick = 31300
element_to_pick = 5

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

def show_FRANKA_pkl_images(file_path):
    # Caricamento dei data
    data = carica_pkl(file_path)

    # Esplora la struttura dei data
    print("Chiavi principali del dizionario:", data[0].keys())
    print("Chiavi di 'observations':", data[0]['observations'].keys())


    # Estrazione dell'immagine e stato da aggiustare a mio caso sim, funziona con file vecchi x ora.. mio file salvato non ho salvato immagini!
    front_data = data[element_to_pick]['observations']['wrist']  # Batch di immagini SOSTITUIRE front / wrist per imagini esterne/dalPolso
    state_data = data[element_to_pick]['observations']['state']  # Batch di stati

    # Assicurati che esistano data di immagini
    if front_data is not None and len(front_data) > 0:
        print("Shape di 'front_data':", front_data.shape) 
        #Shape di 'front_data': (1, 128, 128, 3)

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

    
def show_flattened_and_withBatchSize_UR_pkl_images(file_path):
    # Caricamento dei data
    data = carica_pkl(file_path)

    # Esplora la struttura dei data
    print("Chiavi principali del dizionario:", data[0].keys())
    print("Chiavi di 'observations':", data[0]['observations'].keys())


    # Estrazione dell'immagine e stato da aggiustare a mio caso sim, funziona con file vecchi x ora.. mio file salvato non ho salvato immagini!
    front_data = data[element_to_pick]['observations']['right']  # Batch di immagini SOSTITUIRE front / wrist per imagini esterne/dalPolso
    state_data = data[element_to_pick]['observations']['state']  # Batch di stati

    # Assicurati che esistano data di immagini
    if front_data is not None and len(front_data) > 0:
        print("Shape di 'right_data':", front_data.shape) 

        # Visualizza il primo frame
        plt.imshow(front_data[0])  # Cambia indice per visualizzare altre immagini
        plt.title("Immagine 'RIGHT'")
        plt.colorbar()  # Aggiunge una barra colori
        plt.show()
    else:
        print("Nessun dato di immagine trovato in 'front_data'.")
    

def show_NON_FLATTENED_UR_pkl_images(file_path):
    # Caricamento dei data
    data = carica_pkl(file_path)

    # Esplora la struttura dei data
    print("Chiavi principali del dizionario:", data[0].keys())
    print("Chiavi di 'observations':", data[0]['observations'].keys())

    # Estrazione dell'immagine e stato da aggiustare a mio caso sim, funziona con file vecchi x ora.. mio file salvato non ho salvato immagini!
    right_camera_data = data[element_to_pick]['observations']['images']['right']  # Batch di immagini SOSTITUIRE front / wrist per imagini esterne/dalPolso
    state_data = data[element_to_pick]['observations']['state']  # Batch di stati

    # Assicurati che esistano data di immagini
    if right_camera_data is not None and len(right_camera_data) > 0:
        print("Shape di 'right_camera_data':", right_camera_data.shape)
        # Shape di 'right_camera_data': (240, 320, 3) --> mi sa che devo aggiungere 1° elemento= batch size
        #  per dare img in past al modello di RL (oltre che resizare a 128x128?)


        # Visualizza il primo frame
        # plt.imshow(right_camera_data[0])  # Cambia indice per visualizzare altre immagini
        plt.imshow(right_camera_data)  # Cambia indice per visualizzare altre immagini
        plt.title("Immagine 'right_camera'")
        plt.colorbar()  # Aggiunge una barra colori
        plt.show()
    else:
        print("Nessun dato di immagine trovato in 'right_camera_data'.")


def show_FLATTENED_UR_pkl_images(file_path):
    # Caricamento dei data
    data = carica_pkl(file_path)

    # Esplora la struttura dei data
    print("Chiavi principali del dizionario:", data[0].keys())
    print("Chiavi di 'observations':", data[0]['observations'].keys())

    right_camera_data = data[element_to_pick]['observations']['right']

    # Assicurati che esistano data di immagini
    if right_camera_data is not None and len(right_camera_data) > 0:
        print("Shape di 'right_camera_data':", right_camera_data.shape)

        plt.imshow(right_camera_data)  # Cambia indice per visualizzare altre immagini
        plt.title("Immagine 'right_camera'")
        plt.colorbar()  # Aggiunge una barra colori
        plt.show()
    else:
        print("Nessun dato di immagine trovato in 'right_camera_data'.")


if __name__ == "__main__":
    show_pkl_content(file_path)


    # show_FRANKA_pkl_images(file_path)

    # show_NON_FLATTENED_UR_pkl_images(file_path)
    # show_FLATTENED_UR_pkl_images(file_path)

    show_flattened_and_withBatchSize_UR_pkl_images(file_path)