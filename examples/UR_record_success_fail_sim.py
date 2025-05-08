import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard

# import mujoco.viewer

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

class RecorderNode(Node):
    def __init__(self):
        super().__init__('succ_fail_recorder_node')

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
                self.identical_action_count += 1
            else:
                self.identical_action_count = 0

            # Azzerare i primi tre elementi solo se l'azione è identica per 5 step consecutivi
            if self.identical_action_count >= 5:
                action[:3] = np.zeros(3)
            else:
                # Aggiorna i primi tre elementi dell'ultima azione
                self.last_action[:3] = action[:3]

            self.get_logger().info(f"Action: {action}, Identical Count: {self.identical_action_count}")
        return action
    

FLAGS = flags.FLAGS
# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
# flags.DEFINE_integer("successes_needed", 200, "Number of successful transistions to collect.")
flags.DEFINE_integer("successes_needed", 1, "Number of successful transistions to collect.")
proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"] 


success_key = False
# start_key = False
def on_press(key):
    global success_key#, start_key
    try:
        if str(key) == 'Key.enter':
            success_key = True
        # if str(key) == 'Key.shift':
        #     start_key = True
    except AttributeError:
        pass

def main(_):
    global success_key#, start_key
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    rclpy.init()
    # Create ROS Node
    ros_node = RecorderNode()
    # Start Ros Node on separate thread 
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    env = URPickRosEnv()
    env = RelativeFrame(env) ####### wrapper per convertire observation da frame base a frame "fittizio" = quello iniziale dell'end effector
    env = SERLObsWrapper(env, proprio_keys=proprio_keys) ## wrapper per rendere flattend le observation state
    #############################################
    # VEDI NOTE UR_RECORD_DEMOS_SIM
    ################################################
    # proprio_space = gym.spaces.Dict(
    #     {key: env.observation_space["state"][key] for key in proprio_keys}
    # )
    # print("Proprio Space:", proprio_space)

    obs, _ = env.reset()
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    
    print("press enter to record a successful transition.\n")

    failure_count = 0  # Contatore per le transizioni di failure (MIO TEST ma sarebbe meglio avere n. di steps più sensato)

    while len(successes) <= success_needed:            
        # if start_key:
        actions = ros_node.get_joystick_action()

        next_obs, rew, done, truncated, info = env.step(actions)

        #####################
         # Trasforma l'osservazione iniziale usando il metodo di serl_obs_wrapper.py

        # next_obs = flatten_observations(next_obs, proprio_space, proprio_keys)
        # print("Osservazione appiattita:", next_obs)
        
        #####################
        
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
            )
        obs = next_obs
        if success_key:
            successes.append(transition)
            pbar.update(1)
            success_key = False
        else:
            # qui secondo me con il problema della "quantità" di step che fa nel mio caso
            # avrei 2 problemi per questo passaggio:
            # A) avrei TANTISSIMI Failures registrati (dato che son tutti failures tranne quelli in cui premo tasto)
            # confermo (con 1 solo success fatto velocemente ho 1678 failures)
            # 
            # B) secondo me registrerei anche i casi CORRETTI come FAILURES:
            #  essendo frequentissimo, nei frame subito prima e subito dopo che io prema il tasto per
            # registrare come successo, il codice registrerebbe varie centinaia di failures
            # nella stessa identica situazione (di Obs e actions)
            # 
            # possibili soluzioni:
            # 
            # 1) capire come diminuire il numero di STEP effettuati
            # 
            # 2) implementare logica che ad esempio salva solo un failures ogni tot (100?)
            
            # failures.append(transition)
        
            failure_count += 1 #MIO TEST

            # Salva solo 1 failure ogni 100 
            if failure_count % 25 == 0:
                failures.append(transition)

        if done or truncated:
            obs, _ = env.reset()
            #####################
            # obs = flatten_observations(obs, proprio_space, proprio_keys)
            #####################
        if len(successes) >= success_needed:
            break

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/A_my_UR_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/A_my_UR_failure_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
        
if __name__ == "__main__":
    app.run(main)
