import copy
import os
from tqdm import tqdm
import numpy as np
import pickle as pkl
import datetime
from absl import app, flags
from pynput import keyboard
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64
import threading
from std_srvs.srv import Trigger 

########### gym environment ###########
# from ur_hiro_sim.envs.Ros_UR_PickCube_gym_env import URPickRosEnv
from ur_hiro_sim.envs.TestCamera_Ros_UR_PickCube_gym_env import Real_URPickRosEnv

########### SERL wrappers ###########
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    # MultiCameraBinaryRewardClassifierWrapper,
    UR_GripperPenaltyWrapper,
)
from serl_launcher.wrappers.chunking import ChunkingWrapper

#########################################################
from franka_env.envs.UR_JoystickAction import JoystickInterventionWrapper
import time
#########################################################   


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_boh(x):
    return print("\033[95m {}\033[00m".format(x))

FLAGS = flags.FLAGS # original succ = 200
flags.DEFINE_integer("successes_needed", 150, "Number of successful transistions to collect.")
proprio_keys = ["tcp_pose", "gripper_pose"] 
# proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"] 


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

    # rclpy.init()
    # Create ROS Node
    #####################ros_node = RecorderNode()
    # Start Ros Node on separate thread 
    #####################ros_thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
    #####################ros_thread.start()

    env = Real_URPickRosEnv() # URPickRosEnv
    env = JoystickInterventionWrapper(env) 

    # add wrappers
    env = RelativeFrame(env) # wrapper per convertire observation da frame base a frame "fittizio" = quello iniziale dell'end effector
    env = Quat2EulerWrapper(env) # converte tcp pose rotation da quat a euler
    # commento SerloBs per testare senzxa immagini senno sto wrapper credo generi errore
    env = SERLObsWrapper(env, proprio_keys=proprio_keys) # wrapper per rendere flattend le observation state
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None) # organizza in chunk di dim=1 nel mio caso (resiza anche images con batch size)
    env = UR_GripperPenaltyWrapper(env, penalty=-0.02) # aggiunge penalty per il gripper
   

    #####################ros_node.reset_cmd()  # Reset the recorder node
    obs, _ = env.reset()  # Gym's reset
    successes = []
    failures = []
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    failure_count = 0  # Step counter (to record 1  failure transition per TOT (100-150...) steps)
    successes_count = 0  # succ counter to  reset env after tot successes
    
    print("press enter to record a successful transition.\n")
    
    # while len(successes) < success_needed:            ###### To record SUCCESSES 
    while len(failures) < 400:                          ###### To record FAILURES 
        # if start_key:
        #############################################
        actions = np.zeros(4) # fake policy di zeri
        # actions = env.action_space.sample() 
        #############################################

        ############################################## actions = ros_node.get_joystick_action()

        next_obs, rew, done, truncated, info = env.step(actions)
        if "intervene_action" in info:
                    actions = info["intervene_action"]
        # print("actions: ", actions)
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

            successes_count+= 1
            # manually reset after 15 successes (serve? senno è one-shot...)
            # if successes_count % 25 == 0: 
                # env.reset()
                # ros_node.reset_cmd() 
                # pass
        else:
            failure_count += 1 
            # Register 1 failure transition per TOT steps
            if failure_count % 2 == 0:
                failures.append(transition)
                print_green(f"FAIL N° {failure_count/2} ")
                # print(f" *** Transition OBS STATE: {transition['observations']['state']}")
                # print(failure_count)
                print_boh(f"filures_len = {len(failures)}")


        # commento reset al done tanto qui non serve in teoria
        # if done or truncated:
            # obs, _ = env.reset() # gym's reset
            # #############################################ros_node.reset_cmd()  # Reset the recorder node

        # if len(successes) >= success_needed:
        #     break

        # Sleep for 1 second
        # time.sleep(0.05) ### original
        time.sleep(0.2) ### provo a diminuire freq step robot reale..

    if not os.path.exists("./classifier_data"):
        os.makedirs("./classifier_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./classifier_data/succ/A_EXTRA_Real_W_GripperMounted_{success_needed}_success_images_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(successes, f)
        print(f"saved {success_needed} successful transitions to {file_name}")

    file_name = f"./classifier_data/fails/A_EXTRA_Real_W_GripperMounted_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(failures, f)
        print(f"saved {len(failures)} failure transitions to {file_name}")
        
if __name__ == "__main__":
    app.run(main)








####################################################################################################
####################################################################################################
                                        # 8 AGOSTO
# altezza di Success Cubo faccio intorno a 0.38 ( leggermente più basso di h spawn che è 0.43)


####################################################################################################
####################################################################################################
