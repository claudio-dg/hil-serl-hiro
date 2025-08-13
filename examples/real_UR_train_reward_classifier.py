import glob
import os
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

########### gym environment ###########
from ur_hiro_sim.envs.Ros_UR_PickCube_gym_env import URPickRosEnv

########### SERL wrappers ###########
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    # MultiCameraBinaryRewardClassifierWrapper,
    UR_GripperPenaltyWrapper,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_boh(x):
    return print("\033[95m {}\033[00m".format(x))

from gymnasium import spaces
from gymnasium.spaces import flatten_space, Dict, Box

FLAGS = flags.FLAGS
# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

######### HARD CODED ENV. Configuration values #########
proprio_keys = ["tcp_pose", "gripper_pose"] 

# classifier_keys = ["right"]
# classifier_keys = ["front", "wrist"]
classifier_keys = ["my_realsense","my_basler"]

height: int = 128 # 240
width: int = 128 # 320

action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0]), # x y z traslation + gripper = 4
            high=np.asarray([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

# secondo me qua basta in caso chiamare il mio nuo env, dato che non chiama più mujoco...
proprio_space = spaces.Dict(
    {
        # metto 6tcp pose percho so gia eulero e non quaternions
        "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(1, 6,), dtype=np.float32),
        # "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(1, 6,), dtype=np.float32),
        "gripper_pose": spaces.Box(-1, 1, shape=(1, 1,), dtype=np.float32),
    }
)

flattened_state_space = flatten_space(proprio_space)

image_space = spaces.Dict(
    {
        "my_realsense": spaces.Box(
                                low=0,
                                high=255,
                                shape=(1,height, width, 3),  ######### modificato qua e va??? WTF??? ho aggoiunto 1 all'inizio
                                # secondo me viene giusto perchè la dimensione giusta deve essewre effetticamente con 5 simensioni
                                # B T H W C.. ma sto uno a caso secondo me è sbagliato.. perchè obs space non dovrebbe avarlo...
                                # provare un modo sensato per far venire quello stack a 2?
                                # TEST 1 --Z> vedere se il ckpoint trainato fa qualcosa di sensato -> Si,  TOP
                                
                                #### NOTA #### --> giusto mettere 1 qua !!
                                # Forse perchè effettivamente chunkWrapper modifica Obs_space aggiungendo batch size, e quindi il env.obs_space che
                                # loro prendono in questo script effettivamente ha il batch size = 1 iniziale!!! STO HARD CODDANDO LA COSA...
                                dtype=np.uint8,
                            ),
        "my_basler": spaces.Box(
                                low=0,
                                high=255,
                                shape=(1, height, width, 3), #########
                                dtype=np.uint8,
                            ),
    }
)

observation_space = spaces.Dict(
    {
        "state": flattened_state_space,  # Stato "flattened"
        **image_space,                  # Immagini
    }
)

print("\n\n")
print(observation_space)
print("\n\n")


def main(_):
    devices = jax.local_devices() #Ottiene i dispositivi disponibili (ad esempio, GPU o TPU).
    sharding = jax.sharding.PositionalSharding(devices) # Configura lo "sharding" per distribuire i dati sui dispositivi.
    
    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        observation_space,
        action_space,
        capacity=20000, # Il buffer può contenere fino a 20.000 transizioni.
        include_label=True,
    )

    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data/Real_Robot_Gripper/succ", "*success*.pkl"))
    print(f"success paths: {success_paths}")
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        print_green(f"\n\n\n\n Numero di transizioni in {path}: {len(success_data)}")
        for trans in success_data:
            if "images" in trans['observations'].keys():
                continue

            trans["labels"] = 1
            trans['actions'] = action_space.sample()
###################################################################
            # # FIX: Rimuovi la dimensione batch se presente
            # for k in ["my_basler", "my_realsense"]:
            #     img = trans["observations"][k]
            #     if img.shape[0] == 1 and img.ndim == 4:
            #         trans["observations"][k] = img[0]
###################################################################
            pos_buffer.insert(trans)
            
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )
    
    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        observation_space,
        action_space,
        capacity=50000,
        include_label=True,
    )
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data/Real_Robot_Gripper/fails", "*failure*.pkl"))
    for path in failure_paths:
        failure_data = pkl.load(
            open(path, "rb")
        )
        for trans in failure_data:
            # print_boh(f"----- {trans['observations'].keys()}")
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            trans['actions'] = action_space.sample()
###################################################################
            # FIX: Rimuovi la dimensione batch se presente
            # for k in ["my_basler", "my_realsense"]:
            #     img = trans["observations"][k]
            #     if img.shape[0] == 1 and img.ndim == 4:
            #         trans["observations"][k] = img[0]
###################################################################
            # print_boh(f"Shape delle immagini BAS: {trans["observations"]["my_basler"].shape} ")
            # print_boh(f"Shape delle immagini Realsense: {trans["observations"]["my_realsense"].shape} ")

            neg_buffer.insert(trans)
            # print_green(f" \n \n AAAAAAAAAAAAAAAAAAAAAAAAAAA ")

            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    print_boh(f"failed buffer size: {len(neg_buffer)}")
    print_boh(f"success buffer size: {len(pos_buffer)}")

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)
    
    # print("sample neg pos {}".format(sample))
    
    rng, key = jax.random.split(rng)

    # print_green(f"Shape delle immagini, {sample["observations"]["my_realsense"].shape}" )
    # print_boh(f"Shape delle immagini: {sample["observations"]["my_basler"].shape} ")


    classifier = create_classifier(key, 
                                   sample["observations"], 
                                #    config.classifier_keys,
                                    classifier_keys,
                                   )

    def data_augmentation_fn(rng, observations):
        # for pixel_key in config.classifier_keys:
        for pixel_key in classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        print_green(f" AAAA batch shape {batch["observations"]["my_realsense"].shape}")

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # Sample equal number of positive and negative examples
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        # Merge and create labels
        batch = concat_batches(
            pos_sample, neg_sample, axis=0
        )
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )
            
        rng, key = jax.random.split(rng)

        print_boh(f" BBBB batch shape {batch["observations"]["my_realsense"].shape}")

        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/Real_robot_W_Gripper_EXTRA_Tuned/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    

if __name__ == "__main__":
    app.run(main)