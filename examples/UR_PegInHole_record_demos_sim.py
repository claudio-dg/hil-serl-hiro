import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
import threading
from std_srvs.srv import Trigger 

########### gym environment ###########
from ur_hiro_sim.envs.Ros_UR_PegInHole_gym_env import URPegInHoleRosEnv
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


import os
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))


FLAGS = flags.FLAGS
flags.DEFINE_integer("successes_needed", 30, "Number of successful demos to collect.")
# proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"] 
proprio_keys = ["tcp_pose", "tcp_vel", "tcp_ft"] 

class DemoRecorderNode(Node):
    def __init__(self):
        super().__init__('demo_recorder_node')

        # Reset service to reset gripper commands externally (GUI) coherently to gym's reset
        # TODO: implement service call in complex.cc to reset from GUI
        self.create_service(Trigger, 'reset_recorder', self.reset_callback)

        # Subscriber to 'controller_intervention_offset' topic to receive joystick offsets
        self.offset_subscriber = self.create_subscription(
            Vector3,
            'controller_intervention_offset',
            self.offset_callback,
            10
        )

        # Subscriber to 'controller_intervention_gripper' topic to receive joystick gripper commands
        self.gripper_subscriber = self.create_subscription(
            Float64,
            'controller_intervention_gripper',
            self.gripper_callback,
            10
        )

        # variables to store received inputs  
        self.offset_data = Vector3()
        self.gripper_data = Float64()
        # self.last_action = np.zeros(4)       
        self.last_action = np.zeros(3)       
        self.identical_action_count = 0  # count of action repetitions due to synchronization

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
            # convert joystick data into a numpy array
            # action = np.zeros(4) 
            action = np.zeros(3) 
            action[0] = self.offset_data.x
            action[1] = self.offset_data.y
            action[2] = self.offset_data.z
            # action[3] = self.gripper_data.data

             # Check if offsets are repeated
            if np.array_equal(action[:3], self.last_action[:3]):
                self.identical_action_count += 1
            else:
                self.identical_action_count = 0

            # Set offsets to zero if the same action is repeated for 5 consecutive steps
            if self.identical_action_count >= 5:
                action[:3] = np.zeros(3)
            else:
                self.last_action[:3] = action[:3]
            # self.get_logger().info(f"Action: {action}, Identical Count: {self.identical_action_count}")
        return action
    

    def reset_cmd(self):
        """funzione per resettare lo stato del nodo RecorderNode internamente."""

        self.gripper_data.data = 0.0  # Reset gripper command
        self.offset_data = Vector3()  # Resetta offsets
        # self.last_action = np.zeros(4)  # Resetta last action
        self.last_action = np.zeros(3)  # Resetta last action
        self.identical_action_count = 0  # Reset counter

    def reset_callback(self, request, response):
        """Callback per resettare lo stato del nodo RecorderNode esternamente (es: da GUI)."""

        self.reset_cmd()
        response.success = True
        response.message = "RecorderNode stato resettato con successo."
        self.get_logger().info("Reset del nodo RecorderNode completato.")

        return response

def main(_):

    rclpy.init()
    # Create ROS Node
    ros_node = DemoRecorderNode()
    # Start Ros Node on separate thread 
    ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    ros_thread.start()


    env = URPegInHoleRosEnv() 
    # add wrappers
    env = RelativeFrame(env) # wrapper per convertire observation da frame base a frame "fittizio" = quello iniziale dell'end effector
    env = Quat2EulerWrapper(env) # converte tcp pose rotation da quat a euler
    env = SERLObsWrapper(env, proprio_keys=proprio_keys) # wrapper per rendere flattend le observation state
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None) # organizza in chunk di dim=1 nel mio caso (resiza anche images con batch size)
    # env = UR_GripperPenaltyWrapper(env, penalty=-0.02) # aggiunge penalty per il gripper
   
    #############################################
    # tcp_ft ? NOTA SERLOBSWRAP-> se uso questo però nelle OBS
    # poi ho solo info propriocettive del robot + le immagini, se voglio anche altre info devo modificare
    #  serl_obs_wrapper aggiungendo gli altri campi come fa lui con ** obs_space[images!]
    # per ora provo come viene poi in base a cosa serve modifico! 
    # --> ma se AD es voglio altre info propriocettive come la forza sul tcp, credo basi aggiungerlo alle proprio_keys

    # NOTA: credo poi li prenda in **ordine alfabetico**, 
    # perchè in quello flatten mette per primo
    #  il gripper_pose poi tcpPose e tcpVel
    ################################################

    ros_node.reset_cmd()     # Reset the recorder node
    obs, info = env.reset()  # Gym's reset
    # print("Osservazione restituita da env.reset():", obs)

    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    step_counter = 0  # Step counter (to record 1 step/transition per TOT (100-150...) steps)
    
    while success_count < success_needed:
    
        actions = ros_node.get_joystick_action()
        # print("ACTIONS = ", actions)

        next_obs, rew, done, truncated, info = env.step(actions)
        # print("Osservazione restituita da env.step():", next_obs)

        returns += rew
        step_counter += 1  

        # Register 1 transition per TOT steps
        if step_counter % 10 == 0: # 175 == 0:
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
            # print(f" *** Transition OBS: {transition['observations']}") #includes images
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
            ros_node.reset_cmd()
            #####################
            
    print("### RECORDING COMPLETED ### \n    n. di successi raggiunti =   ", success_count)

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    file_name = f"./demo_data/Peg_Hole/PiH_{success_needed}_demos_{uuid}.pkl"
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
