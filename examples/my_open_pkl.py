import pickle

import numpy as np
import matplotlib.pyplot as plt


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

####################################################################################################
################################ demo data from UR_rec_demos_sim.py ################################

# file_path = "demo_data/pick_cube_sim_30_demos_2024-12-10_16-39-49.pkl"  #  1782 elementi -> questo è 30 DEMOS original sim !
# file_path = "demo_data/NO_PROTECTION_30_demos_2025-07-09_11-25-25.pkl"  # 1247 elementi ## ------------- ##
# file_path = "demo_data/avvitatore_20_demos_2025-09-01_11-44-26.pkl"  


#####################################################################################################
################################ classifier data from UR_rec_success_fail_sim.py ####################

# ****** successi ******


# file_path = "classifier_data/succ/Extra_TEST_SCREWDRIVER_IMG_3_success_images_2025-08-26_11-21-04.pkl"  

# ****** failures ******

file_path = "classifier_data/Avvitatore_well_resized/fails/Wsized_150_TEST_SCREWDRIVER_IMG_failure_imgs_2025-09-03_11-04-05.pkl"  


###################################### TRAINING buffer data ###########################################

# file_path = "NO_PROTECTION_Trainersave_ckpt_Buffer/demo_buffer/transitions_90000.pkl"  # 1247 elementi ## ------------- ##

element_to_pick = 149


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
# 
        print("Tipo del elemento:", type(data[element_to_pick]) , " \n")
        print("Contenuto dell' elemento N:", element_to_pick, " = ", "\n\n ", data[element_to_pick])
        if isinstance(data[element_to_pick], dict):
            print("\n\n Chiavi del dizionario nel primo elemento:", data[element_to_pick].keys(), " \n")
            for key, value in data[element_to_pick].items():
                print(f"- {key}: Tipo = {type(value)}")
                if isinstance(value, np.ndarray):
                    print(f"       Forma = {value.shape}")
                    print(f"       Valori iniziali: {value.flatten()[:5]}")  # Mostra i primi 5 valori
#    
    # Mostra il valore del reward
        if 'rewards' in data[element_to_pick]:
            print(f"\n ## Valore del reward associato alla transizione: {data[element_to_pick]['rewards']}")
    
      # --- NUOVA LOGICA: mostra tutti i valori di "action" ---
        # print_green(f"\nValori del campo 'action' per tutti gli elementi:")
        # for idx, elem in enumerate(data):
        #         print_yellow(f"Elemento {idx}: action = {data[idx]['actions']}")

        # --- NUOVA LOGICA: mostra tutti i valori di "grasp_penalty" ---
        # print_yellow(f"\nValori del campo 'grasp_penalty' per tutti gli elementi:")
        # for idx, elem in enumerate(data):
                # print_yellow(f"Elemento {idx}: grasp_penalty = {data[idx]['grasp_penalty']}")


                
    elif isinstance(data, dict):
        print("\n Chiavi del dizionario:", data.keys())
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

def REAL_show_Basler_E_Dart_pkl_images(file_path):
 
 
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 riga, 2 colonne
    # Caricamento dei data
    data = carica_pkl(file_path)
 
    # Esplora la struttura dei data
    print(" \n\n Chiavi principali del dizionario:", data[0].keys())
    print("Chiavi di 'observations':", data[0]['observations'].keys())
 
 
    # Estrazione dell'immagine e stato da aggiustare a mio caso sim, funziona con file vecchi x ora.. mio file salvato non ho salvato immagini!
    Realsense_data = data[element_to_pick]['observations']['my_realsense']  # Batch di immagini SOSTITUIRE front / wrist per imagini esterne/dalPolso
    state_data = data[element_to_pick]['observations']['state']  # Batch di stati
 
    # Assicurati che esistano data di immagini
    if Realsense_data is not None and len(Realsense_data) > 0:
        print("Shape di 'my_realsense data':", Realsense_data.shape)
 
        print_green(f"Tipo di Realsense data[0]: {type(Realsense_data[0])}")
 
        # plt.imshow(Realsense_data[0])  # Cambia indice per visualizzare altre immagini
        # plt.title("Immagine 'my_realsense'")
        # plt.colorbar()  # Aggiunge una barra colori
        # plt.show()
        axs[0].imshow(Realsense_data[0])
        axs[0].set_title("my_realsense")
        axs[0].axis('off')  # Nasconde gli assi
    else:
        print("Nessun dato di immagine trovato in 'front_data'.")
    
 
    Basler_data = data[element_to_pick]['observations']['my_basler']  # Batch di immagini SOSTITUIRE front / wrist per imagini esterne/dalPolso
     # Assicurati che esistano data di immagini
    if Basler_data is not None and len(Basler_data) > 0:
        print("Shape di 'Basler_data data':", Basler_data.shape)
 
        print_green(f"Tipo di Basler_data[0]: {type(Basler_data[0])}")
 
        # plt.imshow(Basler_data[0])  # Cambia indice per visualizzare altre immagini
        # plt.title("Immagine 'BASLER'")
        # plt.colorbar()  # Aggiunge una barra colori
        # plt.show()
        axs[1].imshow(Basler_data[0])
        axs[1].set_title("my_basler")
        axs[1].axis('off')
 
    else:
        print("Nessun dato di immagine trovato in 'front_data'.")
    plt.tight_layout()
    plt.show()

def show_flattened_and_withBatchSize_UR_pkl_images(file_path):
    # Caricamento dei data
    data = carica_pkl(file_path)

    # Esplora la struttura dei data
    print(" \n\n Chiavi principali del dizionario:", data[0].keys())
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

    # show_flattened_and_withBatchSize_UR_pkl_images(file_path)

    REAL_show_Basler_E_Dart_pkl_images(file_path)