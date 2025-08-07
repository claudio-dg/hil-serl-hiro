import os
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import checkpoints
from tqdm import tqdm
import pickle as pkl
 
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.networks.reward_classifier import create_classifier
from absl import app, flags
from gymnasium.spaces import flatten_space, Dict, Box
 
import collections

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_boh(x):
    return print("\033[95m {}\033[00m".format(x))

FLAGS = flags.FLAGS
# flags.DEFINE_string("test_data_path", "classifier_data/succ/sec_testSet100_succ.pkl", "Path to test data.")
# flags.DEFINE_string("test_data_path", "classifier_data/succ/test_set_circa101Succ.pkl", "Path to test data.")
# flags.DEFINE_string("test_data_path", "classifier_data/succ/AAA_my_UR_200_success_images_2025-05-13_15-49-59.pkl", "Path to test data.")
# flags.DEFINE_string("test_data_path", "classifier_data/succ/A_my_UR_10_succ_images_2025-05-13_14-12-59.pkl", "Path to test data.")
# flags.DEFINE_string("test_data_path", "demo_data/Z_final_my_30_demos_2025-05-20_11-54-40.pkl", "Path to test data.")


# flags.DEFINE_string("test_data_path", "classifier_data/Real_Robot/succ/REAL_Mounted_200_success_images_2025-08-01_17-27-10.pkl", "Path to test data.")
# flags.DEFINE_string("test_data_path", "classifier_data/succ/Z_TestSet_REAL_Mounted_100_success_images_2025-08-05_09-39-49.pkl", "Path to test data.")
flags.DEFINE_string("test_data_path", "classifier_data/fails/ZZZ_TestSet_REAL_Mounted_2025-08-05_09-52-26.pkl", "Path to test data.")

flags.DEFINE_integer("batch_size", 32, "Batch size.")


# Definizione manuale delle chiavi e dello spazio delle osservazioni
# classifier_keys = ["right"]
classifier_keys = ["my_realsense","my_basler"]
height = 128
width = 128
proprio_space = Dict(
    {
        "tcp_pose": Box(-np.inf, np.inf, shape=(1, 6,), dtype=np.float32),
        # "tcp_vel": Box(-np.inf, np.inf, shape=(1, 6,), dtype=np.float32),
        "gripper_pose": Box(-1, 1, shape=(1, 1,), dtype=np.float32),
    }
)
 
flattened_state_space = flatten_space(proprio_space)
 
image_space = Dict(
    {
        # "right": Box(
        #     low=0,
        #     high=255,
        #     shape=(1, height, width, 3), #### qui avevo già messo infatti l'1 aggiuntivo manualmente!!!!
        #     dtype=np.uint8,
        # ),
        "my_realsense": Box(
            low=0,
            high=255,
            shape=(1, height, width, 3), # 1 !!
            dtype=np.uint8,
        ),

        "my_basler": Box(
            low=0,
            high=255,
            shape=(1, height, width, 3), # 1 !!!
            dtype=np.uint8, 
        ),
    }

)
 
observation_space = Dict(
    {
        "state": flattened_state_space,  # Stato "flattened"
        **image_space,                  # Immagini
    }
)
 
def main(_):
    # Carica il modello salvato
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    sample_obs = observation_space.sample()
    classifier = create_classifier(key, sample_obs, classifier_keys)
    classifier = checkpoints.restore_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/Real_robot/"),
        target=classifier,
    )
    print("Modello caricato con successo!")
 
    # Carica i dati di test
    with open(FLAGS.test_data_path, "rb") as f:
        test_data = pkl.load(f)

    print("Tipo test_data:", type(test_data))
    if isinstance(test_data, dict):
        test_data = list(test_data.values())
    elif isinstance(test_data, (list, tuple)):
        pass  # già ok
    elif isinstance(test_data, collections.deque):
        test_data = list(test_data)
    else:
        raise TypeError(f"test_data è di tipo {type(test_data)} e non è gestito automaticamente.")

    for b in test_data:
        # b['labels'] = int(b['rewards'] > 0.75)  # o la soglia che preferisci
        b['labels'] = 1  # cambiare labels se true o false
    
    # Prepara i batch di test
    test_batches = [
        test_data[i : i + FLAGS.batch_size]
        for i in range(0, len(test_data), FLAGS.batch_size)
    ]
    # print(test_data[0])
    # Funzione per calcolare l'accuratezza
    @jax.jit
    def evaluate_step(params, batch):
        logits = classifier.apply_fn(
            {"params": params}, batch["observations"], train=False
        )
        predictions = (nn.sigmoid(logits) >= 0.5).astype(int)
        accuracy = jnp.mean(predictions == batch["labels"])
        return predictions, batch["labels"], accuracy

    # Valutazione del modello
    total_accuracy = 0.0
    all_labels = []
    all_predictions = []
    for batch in tqdm(test_batches, desc="Valutazione"):
        batch = simple_concat_batches(batch)
        predictions, labels, accuracy = evaluate_step(classifier.params, batch)
        print_green(f"Predizioni: {np.array(predictions[:1000]).flatten()} ")
        print_boh(f"Label vere: {np.array(labels[:1000]).flatten()}")
        total_accuracy += accuracy
        all_labels.append(np.array(labels).flatten())
        all_predictions.append(np.array(predictions).flatten())

    total_accuracy /= len(test_batches)
    print_green(f"\n Accuratezza totale del modello: {total_accuracy:.4f}")

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)
    print("\n\n\n\n ###################################### Distribuzione label nel test set:", np.unique(all_labels, return_counts=True))
 
def simple_concat_batches(batch):
    out = {}
    # Gestisci le osservazioni come dict di array
    obs_keys = batch[0]['observations'].keys()
    out['observations'] = {}
    for ok in obs_keys:
        arrs = [np.atleast_1d(b['observations'][ok]) for b in batch]
        out['observations'][ok] = np.stack(arrs, axis=0)
    # Gestisci le labels
    arrs = [np.atleast_1d(b['labels']) for b in batch]
    out['labels'] = np.stack(arrs, axis=0)
    return out
 
if __name__ == "__main__":
    app.run(main)