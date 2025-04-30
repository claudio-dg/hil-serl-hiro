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

# from experiments.mappings import CONFIG_MAPPING
# import mujoco.viewer
from pynput import keyboard

from ur_hiro_sim.envs.Ros_UR_PickCube_gym_env import URPickRosEnv
# fare : export PYTHONPATH=$PYTHONPATH:~/ros/catkin_ws/src/hil-serl/ur_hiro_sim
#  & anche: export PYTHONPATH=$PYTHONPATH:/home/claudiodelgaizo/ros/catkin_ws/src/hil-serl/ur_hiro_sim/ur_hiro_sim/envs
# il secondo serve per wait4message
import os
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))


FLAGS = flags.FLAGS
# flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 2, "Number of successful demos to collect.")

# start_key = False
# def on_press(key):
#     global start_key
#     try:
#         if str(key) == 'Key.shift':
#             start_key = True
#     except AttributeError:
#         pass


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

        # Subscriber per il topic 'mujoco_ros/gripper_command'
        self.gripper_subscriber = self.create_subscription(
            Float64,
            'mujoco_ros/gripper_command',
            self.gripper_callback,
            10
        )

        # Variabili per memorizzare i dati ricevuti
        self.offset_data = Vector3()
        self.gripper_data = Float64()

        # Lock per garantire accesso sicuro ai dati
        self.data_lock = threading.Lock()

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
            self.get_logger().info(f"Action: {action}")
        return action

################### METTERE AZIONE jiystick tramite subscriber qua
def main(_):

    # Inizializza ROS2
    rclpy.init()
    # Crea il nodo ROS
    ros_node = DemoRecorderNode()
    # Avvia il nodo ROS in un thread separato
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()

    
    # global start_key
    # listener = keyboard.Listener(
    #     on_press=on_press)
    # listener.start()
    # assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    # config = CONFIG_MAPPING[FLAGS.exp_name]()

    # env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    env = URPickRosEnv() 
    ######################################################################################################
    # nel mo caso non avendo (X ORA) dei wrapper, prendo direttamente l'env senza passare per il config
    # che appunto è quello che mappa l'env base della task per avvolgerlo con vari wrapper 
    ######################################################################################################

    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    # print("Press shift to start recording.\nIf your controller is not working check controller_type (default is xbox) is configured in examples/experiments/pick_cube_sim/config.py")
    print(" \n\n\n ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ \n\n\n")
    # rate = env._ros_node.create_rate(10)

    # with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    # while viewer.is_running():
    while success_count < success_needed:
    
        # print(" \n\n\n XXX \n\n\n")
        # if start_key:
        # actions = np.zeros(env.action_space.sample().shape) 

        actions = ros_node.get_joystick_action()
        print("ACTIONS = ", actions)
        # se funza vedere come gestire poi alternanza con altra action in futuro (qui non serve perchè unico input è joystick)
        # non funziona.. perchè dal my_open_pkl.py vedo che actions sempre = 0,0,0??
        # OLTRE A DARE 0,0,0 inspiegabilmente, sto print invece mostra altro problema: in assenza di modifiche prende sempre l'ulitmo comando
        # quindi anzichè passare ad action = 0,0,0, ripete al'ifinito l'ultimo ricevuto il che è sbagliato perchè
        # robot lo interpreterà come (ok devo continuare a muovermi in quella direzione)
        # TODO. fix 
        
        next_obs, rew, done, truncated, info = env.step(actions)
        # viewer.sync()
        returns += rew
        # if "intervene_action" in info:
        #     actions = info["intervene_action"]
        transition = copy.deepcopy(
            dict(
                observations=obs, ## questo dovrebbe riempirsi correttamente, MA
                actions=actions, ### secondo me nel mio caso NON metti inserisco MAI azioni QUA!! mette sepre np.zeros
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
                )
        )
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
        # if success_count >= success_needed:
        #     print("\n\n aaaaaaaaaaaaaaaaaa")
        #     break
            
    print("bbbbbbbbb     n. di successi raggiunti =   ", success_count)

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
