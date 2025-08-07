#!/usr/bin/env python3

import glob
import time
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints
import os
################################################
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
### nota: ho visto solo ora che tizi nei .sh dei casi reali es di pick_USB mettono queste due righe!!!

# export XLA_PYTHON_CLIENT_PREALLOCATE=false && \

# export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \ #### questo nell'sh dell'actor
# export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \ #### questo nell'sh del learner!!

# GPU memory allocation --> https://docs.jax.dev/en/latest/gpu_memory_allocation.html

# JAX will preallocate 75% of the total GPU memory when the first JAX operation is run. Preallocating minimizes allocation overhead and memory fragmentation, but can sometimes cause out-of-memory (OOM) errors. If your JAX process fails with OOM, the following environment variables can be used to override the default behavior:

# XLA_PYTHON_CLIENT_PREALLOCATE=false

#     This disables the preallocation behavior. JAX will instead allocate GPU memory as needed, potentially decreasing the overall memory usage. However, this behavior is more prone to GPU memory fragmentation, meaning a JAX program that uses most of the available GPU memory may OOM with preallocation disabled.
# XLA_PYTHON_CLIENT_MEM_FRACTION=.XX

#     If preallocation is enabled, this makes JAX preallocate XX% of the total GPU memory, instead of the default 75%. Lowering the amount preallocated can fix OOMs that occur when the JAX program starts.
# XLA_PYTHON_CLIENT_ALLOCATOR=platform

#     This makes JAX allocate exactly what is needed on demand, and deallocate memory that is no longer needed (note that this is the only configuration that will deallocate GPU memory, instead of reusing it). This is very slow, so is not recommended for general use, but may be useful for running with the minimal possible GPU memory footprint or debugging OOM failures.

################################################
import copy
import pickle as pkl
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from natsort import natsorted

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_single_arm,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

# from experiments.mappings import CONFIG_MAPPING
# import mujoco.viewer

########### gym environment ###########
from ur_hiro_sim.envs.TestCamera_Ros_UR_PickCube_gym_env import Real_URPickRosEnv

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    MultiCameraBinaryRewardClassifierWrapper,
    UR_GripperPenaltyWrapper,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper
from franka_env.envs.UR_JoystickAction import JoystickInterventionWrapper

from serl_launcher.networks.reward_classifier import load_classifier_func


# recorded_demos_path = "demo_data/Z_final_my_30_demos_2025-05-20_11-54-40.pkl" #demo_data/AAA_my_UR_TEST_20_demos_2025-05-13_14-57-38.pkl
recorded_demos_path = "demo_data/REAL_ROBOT_30_demos_2025-08-05_15-24-59.pkl" #
trained_Ckpt_path =  "Real_robot_Training"# "1h30_training_checkpoints"
FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", recorded_demos_path, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", trained_Ckpt_path, "Path to save checkpoints.") # NUOVO TRAINING "per ripartire da prec NO_PROTECTION_Trainersave_ckpt_Buffer"
# flags.DEFINE_string("checkpoint_path", partial_training_path, "Path to resume & save checkpoints.") # RIPRENDERE VECCHIO TRAINING
flags.DEFINE_string("eval_checkpoint_path", trained_Ckpt_path, "my Path to the trained checkpoints.")
# flags.DEFINE_string("eval_checkpoint_path", partial_training_path, "my Path to the trained checkpoints.") # per testare ckpt trainato
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 200, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
)  # debug mode will disable wandb logging


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


batch_size = 64 #32 #64 original
# proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"] 
proprio_keys = ["tcp_pose", "gripper_pose"] 
image_keys = ["my_realsense","my_basler"] ################# DA QUA

setup_mode = "single-arm-learned-gripper"
encoder_type = "resnet-pretrained"
discount=0.97
replay_buffer_capacity = 50000
checkpoint_period = 5000 # 1000 #5000 original
max_steps: int = 1000000
log_period: int = 100
random_steps = 0
buffer_period = 2000
training_starts: int = 100
cta_ratio: int = 2
steps_per_update = 50





def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_orange(x):
    return print("\033[93m {}\033[00m".format(x))

def print_blue(x):
    return print("\033[94m {}\033[00m".format(x))

##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """
    if FLAGS.eval_checkpoint_step:
        # Controlla se è stato specificato uno step (numero) di 
        # checkpoint per la valutazione (--eval_checkpoint_step).
        # Se sì, esegue una valutazione dell'agente 
        # invece di raccogliere dati per il training. (?) --> io lo setto da terminale per sicurezza anzichè da qua
        success_counter = 0
        my_success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            # os.path.abspath(FLAGS.checkpoint_path),
            os.path.abspath(FLAGS.eval_checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        for episode in range(FLAGS.eval_n_trajs):
            obs, _ = env.reset() ##################### provo a vedere se non crasha
            time.sleep(0.2)
            print("\n\n\n\n AAAAAAAAAAAAAAAAAAAAA \n\n\n")
            done = False
            start_time = time.time()
            while not done:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    argmax=False,
                    seed=key
                )
              

                actions = np.asarray(jax.device_get(actions))

                ### azione data da policy
                print_orange(f"AZIONE data in pasto allo step ACTOR = {actions}%")

                next_obs, reward, done, truncated, info = env.step(actions)
                

                obs = next_obs

                if done:
                    if reward:
                        dt = time.time() - start_time
                        time_list.append(dt)
                        print(dt)
                    success_counter += reward

                    success = info['succeed']
                    if success: 
                        my_success_counter += 1
                    
                    # print(reward)
                    # print(f"{success_counter}/{episode + 1}")
                    print_orange(f"final REWARD = {reward}")
                    print_orange(f"success rate = {success_counter}/{episode + 1}")

                    print_green(f"Total successes = {my_success_counter}/{episode + 1}")
                    real_succ_rate = (my_success_counter/(episode +1) )*100
                    print_green(f"Actual successe rate = {real_succ_rate}%")
                time.sleep(0.05) #########################provo A RALLENTARE FREQUENZA STEP



        print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
        print(f"average time: {np.mean(time_list)}")
        return  # after done eval, return and exit
    
    start_step = (
        int(os.path.basename(natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")))[-1])[12:-4]) + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=3000,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    time.sleep(0.2) ##################provo vedere se mpm crasha
    done = False

    # training loop
    timer = Timer()
    running_return = 0.0
    already_intervened = False
    intervention_count = 0
    intervention_steps = 0

    pbar = tqdm.tqdm(range(start_step, max_steps), dynamic_ncols=True)

    # start_time = time.time()  # Tempo iniziale #IOOOOOOO

    for step in pbar:
        timer.tick("total")
         # Stampa il numero di step corrente e il tempo trascorso
        # if step % 10 == 0:  # Stampa ogni 10 step
        #     elapsed_time = time.time() - start_time
        # print(f"Step: {step}, Tempo trascorso: {elapsed_time:.2f} secondi")

        with timer.context("sample_actions"):
            # if step < config.random_steps:
            if step < random_steps:
                actions = env.action_space.sample()
                # print_green(f"(ACTOR) AZIONE .sampled  = {actions}%") # random = 0 ergo non lo fa mai...
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    argmax=False,
                )
                # print_blue(f"(ACTOR) AZIONE PRE jax.device = {actions}%") # anche qui effettivamente rimangono -1 / 0 / 1 !!
                actions = np.asarray(jax.device_get(actions))
                # print_orange(f"(ACTOR) AZIONE POST jax.device = {actions}%")
        # Step environment
        with timer.context("step_env"):
            
            # print_green(f"Azione inviata: {actions}")
            next_obs, reward, done, truncated, info = env.step(actions)

            # if "left" in info:
            #     info.pop("left")
            if "right" in info:
                info.pop("right")

            # override the action with the intervention action
            if "intervene_action" in info:
                actions = info.pop("intervene_action") ####gestire azioni con joystick.. 
                print("intervened!!!")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False

            running_return += reward
            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done,
            )
            if 'grasp_penalty' in info:
                transition['grasp_penalty']= info['grasp_penalty']
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                info["episode"]["intervention_count"] = intervention_count
                info["episode"]["intervention_steps"] = intervention_steps
                stats = {"environment": info}  # send stats to the learner to log
                client.request("send-stats", stats)
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()
                time.sleep(0.2) #########################provo a vedere se non crasha
        # time.sleep(0.25) #########################provo A RALLENTARE FREQUENZA STEP -> con questo registrato 1h30
        time.sleep(0.15) #########################provo A RALLENTARE FREQUENZA STEP

        # if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
        if step > 0 and buffer_period > 0 and step % buffer_period == 0:
            # dump to pickle file
            print_green(f"\n ############# ACTOR step {step}")
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            if not os.path.exists(demo_buffer_path):
                os.makedirs(demo_buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []
            with open(
                os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb"
            ) as f:
                pkl.dump(demo_transitions, f)
                demo_transitions = []

        timer.tock("total")

        # if step % config.log_period == 0:
        if step % log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


##############################################################################


def learner(rng, agent, replay_buffer, demo_buffer, wandb_logger=None):
    """
    The learner loop, which runs when "--learner" is set to True.
    """

    start_step = (
        int(os.path.basename(checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path)))[11:])
        + 1
        if FLAGS.checkpoint_path and os.path.exists(FLAGS.checkpoint_path)
        else 0
    )
    step = start_step

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        # total=config.training_starts,
        total=training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    # while len(replay_buffer) < config.training_starts:
    while len(replay_buffer) < training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    print("Lunghezza buffer dopo primo episodio:", len(replay_buffer))
    
    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")
    # print_green("\n ############# AAAAAAAAAAAAAAAAAAAAAA")

    # 50/50 sampling from RLPD, half from demo and half from online experience
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            # "batch_size": config.batch_size // 2,
            "batch_size": batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={
            # "batch_size": config.batch_size // 2,
            "batch_size": batch_size // 2,
            "pack_obs_and_next_obs": True,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()
    
    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    for step in tqdm.tqdm(
        # range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
        range(start_step, max_steps), dynamic_ncols=True, desc="learner"
    ):
        # print_green("\n  bbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        # for critic_step in range(config.cta_ratio - 1):
        for critic_step in range(cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )
        # publish the updated network
        # if step > 0 and step % (config.steps_per_update) == 0:
        if step > 0 and step % (steps_per_update) == 0:
            print_orange(f"LEARNER STEP: {step}")

            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # if step % config.log_period == 0 and wandb_logger:
        if step % log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        # stampa valore di step
        # print_green(f"\n CCCCCCCCCCCCCCCCCCC ############# step {step}")
        if (
            step > 0
            # and config.checkpoint_period
            and checkpoint_period
            # and step % config.checkpoint_period == 0
            and step % checkpoint_period == 0
        ):
            # print_green(f"\n DDDDDDDDDDDDDDDDDDDdd ############# step {step}")
            
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):

    assert batch_size % num_devices == 0
    # seed
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    # use_trained_reward_classifier = True

    ##########################################################################
    ##### CREO ENVIRONMENT SOLO PER ACTOR, LEARNER NON SERVE (?),.. 
    #  ALTRIMENTI APRO 2 VOLTE TELECAMERE E VIENE GENERATO ERRORE
    # IN REALTÀ LEARNER ACCEDE AD ENV.OBS SPACE AD EsEMPIO
    # QUINDI PROVO A METTERE SOLO LA PARTE DI CLASSIFIER ESCLUSIVA PER L'ACTOR, HA SENSO? PROVO
    ##########################################################################

    # set up the environment
    env = Real_URPickRosEnv(start_camera=FLAGS.actor) ##### attiva camere solo nel caso di ACTOR 
    # add wrappers
    env = JoystickInterventionWrapper(env) 
    env = RelativeFrame(env) # wrapper per convertire observation da frame base a frame "fittizio" = quello iniziale dell'end effector
    env = Quat2EulerWrapper(env) # converte tcp pose rotation da quat a euler
    env = SERLObsWrapper(env, proprio_keys=proprio_keys) # wrapper per rendere flattend le observation state
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None) # organizza in chunk di dim=1 nel mio caso (resiza anche images con batch size)
   
    ################################################################################################################################
    if  FLAGS.actor:
            print_green(f"\n\n ######### ACTOR  ######### \n\n")
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=image_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/Real_robot/"),
            )

            def reward_func(obs, info):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                pred = sigmoid(classifier(obs))
                if int(pred[0] > 0.75):  #### scatola rovinata.. abbasso threshold da 0.85 a 0.75                  
                    print_green(f"prediction del classifier = {sigmoid(classifier(obs))}")
                else:
                    print_blue(f"prediction del classifier = {sigmoid(classifier(obs))}")

                if (info["is_low_enough"]):                    
                    print_green(f"TCP < 0,26 = {info["is_low_enough"]}")
                else:
                    print_blue(f"TCP < 0,26 = {info["is_low_enough"]}")

                return int(pred[0] > 0.75 and info["is_low_enough"]) # obs["state"][0,3] è altezza tcp rispetto a pu nto inizilae (parte da 0 e positivo verso basso ->  > 0.14 corrispnde ad altezza assoluta < 0.26 del TCP)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
    ################################################################################################################################ 
    
    env = UR_GripperPenaltyWrapper(env, penalty=-0.02) # aggiunge penalty per il gripper
    # wrapper del trainer
    env = RecordEpisodeStatistics(env)

    rng, sampling_rng = jax.random.split(rng)
        
    if setup_mode == 'single-arm-fixed-gripper' or setup_mode == 'dual-arm-fixed-gripper':   
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=image_keys,
            encoder_type=encoder_type,
            discount=discount,
        )
        include_grasp_penalty = False

    elif setup_mode == 'single-arm-learned-gripper':
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=image_keys,
            encoder_type=encoder_type,
            discount=discount,
        )
        include_grasp_penalty = True

    # elif config.setup_mode == 'dual-arm-learned-gripper':
    #     agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
    #         seed=FLAGS.seed,
    #         sample_obs=env.observation_space.sample(),
    #         sample_action=env.action_space.sample(),
    #         image_keys=config.image_keys,
    #         encoder_type=config.encoder_type,
    #         discount=config.discount,
    #     )
    #     include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {setup_mode}")

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(
        jax.tree_util.tree_map(jnp.array, agent), sharding.replicate()
    )

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        input("Checkpoint path already exists. Press Enter to resume training.")
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        ckpt_number = os.path.basename(
            checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        )[11:]
        print_green(f"###################### \n Loaded previous checkpoint at step {ckpt_number}. \n###################### ")

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=replay_buffer_capacity,
            image_keys=image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )
        # set up wandb and logging
        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=FLAGS.exp_name,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=replay_buffer_capacity,
            image_keys=image_keys,
            include_grasp_penalty=include_grasp_penalty,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            with open(path, "rb") as f:
                # stampa il path
                print(" path da cui prendo trans per DEMO BUFFER:   ", path)
                transitions = pkl.load(f)
                for transition in transitions:
                    if 'infos' in transition and 'grasp_penalty' in transition['infos']: #infos con la s _> c'è solo in demo e non in suc/fail
                        transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in glob.glob(
                os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")
            ):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        # learner loop
        print_green("starting learner loop")
        learner(
            sampling_rng,
            agent,
            replay_buffer,
            demo_buffer=demo_buffer,
            wandb_logger=wandb_logger,
        )

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)  # the queue size on the actor
        intvn_data_store = QueuedDataStore(50000)

        # actor loop
        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)
