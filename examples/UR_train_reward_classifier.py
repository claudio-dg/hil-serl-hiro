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

# from experiments.mappings import CONFIG_MAPPING
########### gym environment ###########
from ur_hiro_sim.envs.Ros_UR_PickCube_gym_env import URPickRosEnv
# fare : export PYTHONPATH=$PYTHONPATH:~/ros/catkin_ws/src/hil-serl/ur_hiro_sim
#  & anche: export PYTHONPATH=$PYTHONPATH:/home/claudiodelgaizo/ros/catkin_ws/src/hil-serl/ur_hiro_sim/ur_hiro_sim/envs
# il secondo serve per wait4message

########### SERL wrappers ###########
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    # MultiCameraBinaryRewardClassifierWrapper,
    UR_GripperPenaltyWrapper,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper

from gymnasium import spaces
from gymnasium.spaces import flatten_space, Dict, Box

FLAGS = flags.FLAGS
# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 150, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")

######### HARD CODED ENV. Configuration values #########
proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"] 
classifier_keys = ["right"]
height: int = 128 # 240
width: int = 128 # 320
# classifier_keys = ["front", "wrist"]

# observation_space = spaces.Dict(
#                 {
#                     "state": spaces.Dict(
#                         {
#                             "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
#                             "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
#                             "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
#                         }
#                     ),
#                     "images": spaces.Dict(
#                         {
#                             "left": spaces.Box(
#                                 low=0,
#                                 high=255,
#                                 shape=(height, width, 3),
#                                 dtype=np.uint8,
#                             ),
#                             "right": spaces.Box(
#                                 low=0,
#                                 high=255,
#                                 shape=(height, width, 3),
#                                 dtype=np.uint8,
#                             ),
#                         }
#                     ),
#                 }
#     )

action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0]), # x y z traslation + gripper = 4
            high=np.asarray([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

proprio_space = spaces.Dict(
    {
        # metto 6tcp pose percho so gia eulero e non quaternions
        "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(1, 6,), dtype=np.float32),
        "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(1, 6,), dtype=np.float32),
        "gripper_pose": spaces.Box(-1, 1, shape=(1, 1,), dtype=np.float32),
    }
)

flattened_state_space = flatten_space(proprio_space)

image_space = spaces.Dict(
    {
        "right": spaces.Box(
            low=0,
            high=255,
            shape=(1, height, width, 3),
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

# Funzione per replicare lo spazio con obs_horizon (da chunkWrapper)
# def space_stack(space, repeat):
#     if isinstance(space, Box):
#         return Box(
#             low=np.repeat(space.low[None], repeat, axis=0),
#             high=np.repeat(space.high[None], repeat, axis=0),
#             dtype=space.dtype,
#         )
#     elif isinstance(space, Dict):
#         return Dict({k: space_stack(v, repeat) for k, v in space.spaces.items()})
#     else:
#         raise TypeError(f"Unsupported space type: {type(space)}")

# # Imposta il valore di obs_horizon
# obs_horizon = 1

# # Applica la trasformazione
# chunked_observation_space = space_stack(observation_space, obs_horizon)
# print("\n\n")
# print(chunked_observation_space)
print("\n\n")
print(observation_space)
print("\n\n")


def main(_):
    # assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    # config = CONFIG_MAPPING[FLAGS.exp_name]()
    # env = config.get_environment(fake_env=True, save_video=False, classifier=False)


    
    # MI SEMBRA SERVA SOLO A ESTRARRE env.action_space & env.observation_space, quindi
    # provo ad hard coddare queste due vairiabili e togliere l'env --> nota che i wrapper lo modificano quindi devo a
    # aggiungere le modifiche a obs space che fanno i wrapper


    # env = URPickRosEnv() 
    # env = RelativeFrame(env) # wrapper per convertire observation da frame base a frame "fittizio" = quello iniziale dell'end effector
    # env = Quat2EulerWrapper(env) # converte tcp pose rotation da quat a euler
    # env = SERLObsWrapper(env, proprio_keys=proprio_keys) # wrapper per rendere flattend le observation state
    # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None) # organizza in chunk di dim=1 nel mio caso (resiza anche images con batch size)
    # env = UR_GripperPenaltyWrapper(env, penalty=-0.02) # aggiunge penalty per il gripper
   

    devices = jax.local_devices() #Ottiene i dispositivi disponibili (ad esempio, GPU o TPU).
    sharding = jax.sharding.PositionalSharding(devices) # Configura lo "sharding" per distribuire i dati sui dispositivi.
    
    # Create buffer for positive transitions
    pos_buffer = ReplayBuffer(
        # env.observation_space,
        # env.action_space,
        observation_space,
        action_space,
        capacity=20000, # Il buffer può contenere fino a 20.000 transizioni.
        include_label=True,
    )

    # success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data/succ", "*success*.pkl"))
    print(f"success paths: {success_paths}")
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        print(f"\n\n\n\n Numero di transizioni in {path}: {len(success_data)}")
        for trans in success_data:
            # if "right" in trans['observations'].keys():  
            if "images" in trans['observations'].keys(): # MA PERCHÈ CERCA CHIAVE IMAGES? IN TEORIA 
                # DOPO FLATTEN NON ESISTE PIU QUELLA CHIAVE MANCO PER LORO, HANNO 'WRIST'
                continue
            trans["labels"] = 1
            # trans['actions'] = env.action_space.sample()
            trans['actions'] = action_space.sample()
            pos_buffer.insert(trans)
            
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )
    
    # Create buffer for negative transitions
    neg_buffer = ReplayBuffer(
        # env.observation_space,
        # env.action_space,
        observation_space,
        action_space,
        capacity=50000,
        include_label=True,
    )
    # failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data/fails", "*failure*.pkl"))
    for path in failure_paths:
        failure_data = pkl.load(
            open(path, "rb")
        )
        for trans in failure_data:
            if "images" in trans['observations'].keys():
            # if "right" in trans['observations'].keys():
                continue
            trans["labels"] = 0
            # trans['actions'] = env.action_space.sample()
            trans['actions'] = action_space.sample()
            neg_buffer.insert(trans)
            
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,
        },
        device=sharding.replicate(),
    )

    print(f"failed buffer size: {len(neg_buffer)}")
    print(f"success buffer size: {len(pos_buffer)}")

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    rng, key = jax.random.split(rng)
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
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )
    

if __name__ == "__main__":
    app.run(main)