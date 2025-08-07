import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
import requests
from scipy.spatial.transform import Rotation as R
from typing import List
import inputs
import threading
from dataclasses import dataclass
from enum import Enum

import rclpy
import rclpy.executors
from rclpy.node import Node, Executor
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, Vector3
import threading
from ur_hiro_sim.ROS_mujoco_gym_env import MujocoGymEnv 

def print_orange(x):
    return print("\033[93m {}\033[00m".format(x))

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_blu(x):
    return print("\033[94m {}\033[00m".format(x))

class JoystickInterventionWrapper(gym.ActionWrapper):
    
    def __init__(self, env: MujocoGymEnv):
        super().__init__(env)

        # Inizializza il nodo ROS solo se non è già stato inizializzato
        if rclpy.ok() == False:
            rclpy.init()
        
        # Subscriber al topic 'controller_intervention_gripper'
        env._ros_node.create_subscription(
            Float64,
            'controller_intervention_gripper',
            self.gripper_callback,
            1
        )

        # Subscriber al topic 'controller_intervention_offset'
        env._ros_node.create_subscription(
            Vector3,
            'controller_intervention_offset',
            self.offset_callback,
            1
        )

        self.logger = env._ros_node.get_logger()

        # Variabile per il comando del gripper
        self.gripper_command =  Float64()
        self.offset_data = Vector3()
        self.last_action = np.zeros(4)       
        self.identical_action_count = 0  # count of action repetitions due to synchronization

        self.new_joystick_input = False 

        # Contatore per l'azione del joystick 
        self.joystick_action_counter = 0
        self.joystick_action_limit = 3 # consecutive step to mantain action for

    def gripper_callback(self, msg):
        """Callback per aggiornare il comando del gripper."""
        # with self.data_lock:
        self.gripper_command = msg
        self.new_joystick_input = True # segnala nuovi input 
        self.logger.info(f"Ricevuto comando gripper: {msg.data}")

    def offset_callback(self, msg):
        """Callback per aggiornare l'offset del joystick."""
        # with self.data_lock:
        self.offset_data = msg
        self.new_joystick_input = True 

    def get_joystick_action(self):
        """Restituisce i dati ricevuti dai subscriber come array NumPy."""
        # convert joystick data into a numpy array
        action = np.zeros(4) 
        action[0] = self.offset_data.x
        action[1] = self.offset_data.y
        action[2] = self.offset_data.z
        action[3] = self.gripper_command.data
        
        ####################################################################################################################
        ################# per fase finale del training provo a forzare chiuso quando iput joystick=RT=SALI #################
        # altrimenti non riesco a fare fare step finale al robot perchè mandando SALI mando sempre anche apertura gripper
        
        # if self.offset_data.z >= 0.05:
            # action[3] = 239.0

        ####################################################################################################################
        ####################################################################################################################
        
        self.last_action = action
        return action

    def reset_cmd(self):
        """funzione per resettare lo stato del nodo RecorderNode internamente."""

        self.gripper_command.data = 0.0  # Reset gripper command
        self.offset_data = Vector3()  # Resetta offsets
        self.last_action = np.zeros(4)  # Resetta last action
        self.identical_action_count = 0  # Reset counter


    def action(self, action):
        """
        Input:
        - action: policy action
        Output:
        - action: joystick action if nonzero; else, policy action
        """

        if self.new_joystick_input:
            
            intervened_action = self.get_joystick_action()
            self.new_joystick_input = False
            self.joystick_action_counter = 0  # Reset counter
            return intervened_action, True

        # Use action for multiple steps (useful in case of step sinch issues)
        # if self.joystick_action_counter < self.joystick_action_limit:
            # self.joystick_action_counter += 1
            # print(f"[DEBUG] Mantieni l'ultima azione del joystick per {self.joystick_action_counter}/{self.joystick_action_limit} step.")
            # return self.last_action, True

        self.reset_cmd()  # Reset the command if no new input is received
        
        # No intervention. Returning policy action.
        ###  Remap gripper policy 1=239  -1 = 0 0=tieni ultimo valore?
        fixPolicy_action = np.array(action, copy=True)
        if action[3] == 1.0:
            fixPolicy_action[3] = 239
        elif action[3] == -1.0:
            fixPolicy_action[3] = 0

        # elif action[3] == 0.0: 
        #     action[3] = ??? come gli dico di ignorare e prendere prec? perchè senno APRE quando manda zero
        # capire quando viene mandato 0
        # print_green(f"[DEBUG] AZIONE POLICY PUBBLICATA: {fixPolicy_action}")

        # re-inserisco cut policy action x test robot reale
        fixPolicy_action[0] *= 0.35
        fixPolicy_action[1] *= 0.35
        fixPolicy_action[2] *= 0.35

        return fixPolicy_action, False
 
    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info

    def close(self):
        self.running = False
        self.thread.join()
        super().close()