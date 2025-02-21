import pickle

# Percorso al file .pkl salvato
file_path = "classifier_data/pick_cube_sim_4_success_images_2024-12-11_09-35-44.pkl"

# Carica il file
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Verifica il tipo e il contenuto
print("Tipo del contenuto:", type(data))
if isinstance(data, list):
    print(f"Il file contiene una lista con {len(data)} elementi.")
    # Analizza il primo elemento
    print("Tipo del primo elemento:", type(data[0]))
    print("Contenuto del primo elemento:", data[0])
    if isinstance(data[0], dict):
        print("Chiavi del dizionario nel primo elemento:", data[0].keys())
        for key, value in data[0].items():
            print(f"- {key}: Tipo = {type(value)}")
            if isinstance(value, np.ndarray):
                print(f"  Forma = {value.shape}")
                print(f"  Valori iniziali: {value.flatten()[:5]}")  # Mostra i primi 5 valori
elif isinstance(data, dict):
    print("Chiavi del dizionario:", data.keys())
else:
    print("Contenuto:", data)

