import pickle

import numpy as np
import matplotlib.pyplot as plt


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


def print_yellow(x):
    return print("\033[93m {}\033[00m".format(x))

####################################################################################################
################################ demo data from UR_rec_demos_sim.py ################################

# file_path = "demo_data/pick_cube_sim_30_demos_2025-03-18_15-46-36.pkl"  # 69 elementi --> ma non è realmente 30 demo mi sa
# file_path = "demo_data/pick_cube_sim_30_demos_2025-03-26_15-01-00.pkl"  # 51 elementi
# file_path = "demo_data/pick_cube_sim_30_demos_2024-12-10_16-39-49.pkl"  #  1782 elementi -Z questo è 30 DEMOS !

# file_path = "demo_data/A_my_UR_TEST_2_demos_first_completed.pkl" # 31316 elementi (?????)



# file_path = "demo_data/AAA_my_UR_TEST_20_demos_2025-05-13_14-57-38.pkl"  # 86 elementi ## ------------- ##
# prima demo va da 0-13 
# seconda demo va da 14 -23
## nota --> in realtà son tipo 18 demo perchè bug per cui a volte quando termina episodio salva due volte di fila 
#  RIFARE, PERCHÈ IN REALTÀ IL TEST CON 30 DEMOS HA CIRCA 1800 TRANS, QUINDI RIFARE AUMENTANDO FREWUNZA CAMPIONAMENTO PER AVERE CIRCA 2K

# file_path = "demo_data/NO_PROTECTION_30_demos_2025-07-09_11-25-25.pkl"  # 1247 elementi ## ------------- ##

#################################################################################################################
################################ classifier data from UR_rec_success_fail_sim.py ################################

# ****** successi ******
# file_path = "classifier_data/pick_cube_sim_4_success_images_2024-12-11_09-35-44.pkl" 
# file_path = "classifier_data/A_my_UR_1_success_images_2025-05-06_14-45-52.pkl"  
# file_path = "classifier_data/A_my_UR_30_success_Con_IMMAGINI_2025-05-07_11-00-17.pkl"  

# file_path = "classifier_data/succ/sec_testSet100_succ.pkl" # giusto 100 elementi
# file_path = "classifier_data/succ/AAA_my_UR_200_success_images_2025-05-13_15-49-59.pkl" # giusto 200 elementi

# 0-24 ok
# 25-49 ok

# ****** failures ******
# file_path = "classifier_data/A_my_UR_failure_images_2025-05-06_14-45-52.pkl" # 1678 elementi
# file_path = "classifier_data/A_my_UR_failure_images_1ogni100failures.pkl" # 10 elementi
# file_path = "classifier_data/A_my_UR_failure_Con_IMMAGINI_2025-05-07_11-00-17.pkl" # 28 elementi
# file_path = "classifier_data/A_my_UR_failure_images_2025-05-08_17-05-05.pkl" # 10 elementi

# file_path = "classifier_data/fails/AAA_my_UR_failure_images_2025-05-13_15-49-59.pkl" # 107 elementi, non abbastanza (suggeriscino 2x/3x rispetto ai succ)
# PROVO A PRENDERE I FAILURES IN TEST SEPARATO.. forse ha anche senso csoì non ho  successi registrati come failures

# file_path = "classifier_data/fails/AAA_my_UR_failure_images_2025-05-13_16-17-23.pkl" # 651 elementi,  test seprato da qui prendo solo failures ## ------------- ##
# element_to_pick = 31300

# file_path = "classifier_data/succ/testWITH_Imgs_real_UR_10_success_images_2025-07-25_14-14-55.pkl"  # 
# file_path = "classifier_data/succ/testWITH_Imgs_real_UR_10_success_images_2025-07-25_16-17-39.pkl"  # 1247 elementi ## ------------- ##
################################################# TRAINING buffer data #####################################################

# file_path = "classifier_data/succ/REALrobot_Imgs_real_UR_10_success_images_2025-07-31_10-15-09.pkl"  # 1247 elementi ## ------------- ##
# file_path = "NO_PROTECTION_Trainersave_ckpt_Buffer/demo_buffer/transitions_90000.pkl"  # 1247 elementi ## ------------- ##


# file_path = "classifier_data/succ/ROBOT_MOUNTED_2025-08-01_16-45-32.pkl"  # 
# file_path = "classifier_data/fails/testNO_Imgs_real_UR_failure_images_2025-08-01_16-45-32.pkl"  #  7 elementi

# file_path = "classifier_data/fails/testNO_Imgs_real_UR_failure_images_2025-07-25_16-17-39.pkl"  # 

########################à REAL ROBOT ########################
# file_path = "classifier_data/fails/REAL_Mounted_2025-08-01_17-27-10.pkl"  #  43 elementi a caso da riprendere
# file_path = "classifier_data/Real_Robot/succ/REAL_Mounted_200_success_images_2025-08-01_17-27-10.pkl"  #  201
# file_path = "classifier_data/Real_Robot/fails/REAL_Mounted_failure_2025-08-04_10-27-02.pkl"  #  1000

file_path = "demo_data/REAL_ROBOT_30_demos_2025-08-05_15-24-59.pkl"  #  2761 elementi

element_to_pick = 2740

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
    # Mostra il valore del reward
            if 'rewards' in data[element_to_pick]:
                print(f"\n ## Valore del reward associato alla transizione: {data[element_to_pick]['rewards']}")
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