import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

###########
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
import threading
###########

from ur_hiro_sim.envs.Ros_UR_PickCube_gym_env import URPickRosEnv
# fare : export PYTHONPATH=$PYTHONPATH:~/ros/catkin_ws/src/hil-serl/ur_hiro_sim
#  & anche: export PYTHONPATH=$PYTHONPATH:/home/claudiodelgaizo/ros/catkin_ws/src/hil-serl/ur_hiro_sim/ur_hiro_sim/envs
# il secondo serve per wait4message

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
# from serl_launcher.wrappers.serl_obs_wrappers import flatten_observations
# import gymnasium as gym
from franka_env.envs.relative_env import RelativeFrame



import os
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))


FLAGS = flags.FLAGS
# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 2, "Number of successful demos to collect.")
proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"] 

class DemoRecorderNode(Node):
    def __init__(self):
        super().__init__('demo_recorder_node')

        # Subscriber per il topic 'controller_intervention_offset'
        self.offset_subscriber = self.create_subscription(
            Vector3,
            'controller_intervention_offset',
            self.offset_callback,
            10
        )

        # Subscriber per il topic 'controller_intervention_gripper' che arriva diretto dal joystick
        self.gripper_subscriber = self.create_subscription(
            Float64,
            'controller_intervention_gripper',
            self.gripper_callback,
            10
        )

        # Variabili per memorizzare i dati ricevuti
        self.offset_data = Vector3()
        self.gripper_data = Float64()

        # Lock per garantire accesso sicuro ai dati
        self.data_lock = threading.Lock()

        # Variabile per memorizzare l'ultima azione
        self.last_action = np.zeros(4)  # Inizializza con un array di zeri       
        self.identical_action_count = 0  # conta n di ripetizioni della stessa azione

    def offset_callback(self, msg):
        """Callback per il topic 'controller_intervention_offset'."""
        with self.data_lock:
            self.offset_data = msg
        # self.get_logger().info(f"Ricevuto offset: {msg}")

    def gripper_callback(self, msg):
        """Callback per il topic 'mujoco_ros/gripper_command'."""
        with self.data_lock:
            self.gripper_data = msg
        self.get_logger().info(f"Ricevuto comando gripper: {msg}")

    def get_joystick_action(self):
        """Restituisce i dati ricevuti dai subscriber come array NumPy."""
        with self.data_lock:
            # Converte i dati ricevuti in un array NumPy
            action = np.zeros(4)  # Inizializza un array di 4 elementi
            action[0] = self.offset_data.x
            action[1] = self.offset_data.y
            action[2] = self.offset_data.z
            action[3] = self.gripper_data.data

             # Controlla se i primi tre elementi dell'azione sono identici alla precedente
            if np.array_equal(action[:3], self.last_action[:3]):
                # Incrementa il contatore per azioni identiche
                self.identical_action_count += 1
            else:
                # Resetta il contatore se l'azione è diversa
                self.identical_action_count = 0

            # Azzerare i primi tre elementi solo se l'azione è identica per 5 step consecutivi
            if self.identical_action_count >= 5:
                action[:3] = np.zeros(3)
            else:
                # Aggiorna i primi tre elementi dell'ultima azione
                self.last_action[:3] = action[:3]

            # Mantieni il valore precedente del gripper se non ci sono nuovi comandi (ridondante secondo me)
            # if action[3] == 0.0:  # Supponendo che 0.0 sia il valore di default per il gripper
            #     action[3] = self.last_action[3]
            # else:
            #     # Aggiorna il valore del gripper nell'ultima azione
            #     self.last_action[3] = action[3]

            # self.get_logger().info(f"Action: {action}, Identical Count: {self.identical_action_count}")
        return action

def main(_):

    rclpy.init()
    # Create ROS Node
    ros_node = DemoRecorderNode()
    # Start Ros Node on separate thread 
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()


    env = URPickRosEnv() 
    env = RelativeFrame(env) ####### wrapper per convertire observation da frame base a frame "fittizio" = quello iniziale dell'end effector
    env = SERLObsWrapper(env, proprio_keys=proprio_keys) ## wrapper per rendere flattend le observation state
    #############################################
    # tcp_ft ? NOTA SERLOBSWRAP-> se uso questo però nelle OBS
    # poi ho solo info propriocettive del robot + le immagini, se voglio anche altre info devo modificare
    #  serl_obs_wrapper aggiungendo gli altri campi come fa lui con ** obs_space[images!]
    # per ora provo come viene poi in base a cosa serve modifico!

    # NOTA: credo poi li prenda in **ordine alfabetico**, 
    # perchè in quello flatten mette per primo
    #  il gripper_pose poi tcpPose e tcpVel
    ################################################
    # proprio_space = gym.spaces.Dict(
    #     {key: env.observation_space["state"][key] for key in proprio_keys}
    # )
    # print("Proprio Space:", proprio_space)

    obs, info = env.reset()
    print("Osservazione restituita da env.reset():", obs)

    # obs = flatten_observations(obs, proprio_space, proprio_keys) # NO perchè prende obs= null all'inizio e crasha

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    step_counter = 0  # Contatore per gli step (per  registrare una transition (step) ogni tot (100))
    
    while success_count < success_needed:
    
        actions = ros_node.get_joystick_action()
        # print("ACTIONS = ", actions)

        next_obs, rew, done, truncated, info = env.step(actions)
        # print("Osservazione restituita da env.step():", next_obs)

        #####################
         # Trasforma l'osservazione iniziale usando il metodo di serl_obs_wrapper.py

        # next_obs = flatten_observations(next_obs, proprio_space, proprio_keys)
        # print("Osservazione appiattita:", next_obs)
        
        #####################


        returns += rew
        step_counter += 1  # Incrementa il contatore degli step

        # Registra la transizione solo ogni 100 step
        if step_counter % 200 == 0:
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                    infos=info,
                )
            )
            print(f" *** Transition actions: {transition['actions']}")
            print(f" *** Transition OBS STATE: {transition['observations']['state']}")
            # print(f" *** Transition OBS: {transition['observations']}")
            trajectory.append(transition)
                
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done:
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                print(f"Success count: {success_count}")
                pbar.update(1)
            else:
                print("\n\n  tentativo FALLITO (probabile TIMEOUT)") 
            trajectory = []
            returns = 0           
            obs, info = env.reset()
            #####################
            # obs = flatten_observations(obs, proprio_space, proprio_keys)
            #####################

            
    print("### RECORDING COMPLETED ### \n    n. di successi raggiunti =   ", success_count)

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    file_name = f"./demo_data/A_my_UR_TEST_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

    env.close()
    ros_node.destroy_node()
    rclpy.shutdown()
    ros_thread.join()

def new_func():
    return False

if __name__ == "__main__":
        app.run(main)
