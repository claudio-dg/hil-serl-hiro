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

class JoystickInterventionWrapper(gym.ActionWrapper):
    
    def __init__(self, env):
        super().__init__(env)

        # Inizializza il nodo ROS solo se non è già stato inizializzato
        if rclpy.ok() == False:
            rclpy.init()
        
        self.ros_node = JoystickSubscriberNode(self)

        # Spinning the node in a different thread
        self._executor = rclpy.executors.MultiThreadedExecutor()
        self._executor.add_node(self.ros_node)
        self._spinning_thread = threading.Thread(target=self.spinning_cb)
        self._spinning_thread.start()

        # Contatore per l'azione del joystick 
        self.joystick_action_counter = 0
        self.joystick_action_limit = 25  # Numero di step consecutivi per mantenere l'azione del joystick

    def spinning_cb(self):
        while rclpy.ok():
            self.ros_node.get_logger().info("Started wrPPER NODNENDOENde")
            self._executor.spin()
        self.ros_node.get_logger().info("Finished spinning base gym node")

    def action(self, action):
        """
        Input:
        - action: policy action
        Output:
        - action: joystick action if nonzero; else, policy action
        """
        
        # with self.ros_node.data_lock:
        if self.ros_node.new_joystick_input: # Controlla se ci sono nuovi input dal joystick
            
            intervened_action = self.ros_node.get_joystick_action()
            self.ros_node.new_joystick_input = False
            #########################
            self.joystick_action_counter = 0  # Resetta il contatore
            #########################
            # print("[DEBUG] Nuovo input joystick ricevuto:", intervened_action)
            return intervened_action, True

        #########################
        # Se non ci sono nuovi input, usa l'ultima azione del joystick per un massimo di joystick_action_limit step
        if self.joystick_action_counter < self.joystick_action_limit:
            self.joystick_action_counter += 1
            # print(f"[DEBUG] Mantieni l'ultima azione del joystick per {self.joystick_action_counter}/{self.joystick_action_limit} step.")
            return self.ros_node.last_action, True
        #########################
        ## dovrebbere resettare gripper a 0 per azioni successive
        self.ros_node.reset_cmd()  # Reset the command if no new input is received
    
        
        # print("[ZZZ DEBUG] No intervention. Returning policy action.")  # Debug print
        ### gestisco input gripper policy 1=239  -1 = 0 0=tieni ultimo valore?
        fixPolicy_action = np.array(action, copy=True)  # Crea una copia modificabile
        if action[3] == 1.0:
            fixPolicy_action[3] = 239
        elif action[3] == -1.0:
            fixPolicy_action[3] = 0
        # elif action[3] == 0.0: 
        #     action[3] = ??? come gli dico di ignorare e prendere prec? perchè senno APRE quando manda zero
        # capire quando viene mandato 0
        return fixPolicy_action, False
 
    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        # info["left"] = self.left
        # info["right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        self.running = False
        self.thread.join()
        super().close()

class JoystickSubscriberNode(Node):
    def __init__(self, wrapper):
        super().__init__('joystick_subscriber_node')

        # Subscriber al topic 'controller_intervention_gripper'
        self.create_subscription(
            Float64,
            'controller_intervention_gripper',
            self.gripper_callback,
            10
        )

        # Subscriber al topic 'controller_intervention_offset'
        self.create_subscription(
            Vector3,
            'controller_intervention_offset',
            self.offset_callback,
            10
        )

        # Variabile per il comando del gripper
        self.gripper_command =  Float64()
        self.offset_data = Vector3()
        self.last_action = np.zeros(4)       
        self.identical_action_count = 0  # count of action repetitions due to synchronization

        self.new_joystick_input = False  # Flag per monitorare nuovi input
        self.data_lock = threading.Lock()

    def gripper_callback(self, msg):
        """Callback per aggiornare il comando del gripper."""
        # with self.data_lock:
        self.gripper_command = msg
        self.new_joystick_input = True # segnala nuovi input 
        self.get_logger().info(f"Ricevuto comando gripper: {msg.data}")

    def offset_callback(self, msg):
        """Callback per aggiornare l'offset del joystick."""
        # with self.data_lock:
        self.offset_data = msg
        self.new_joystick_input = True # segnala nuovi input
        # self.get_logger().info(f"Ricevuto offset: x={msg.x}, y={msg.y}, z={msg.z}")

    def get_joystick_action(self):
        """Restituisce i dati ricevuti dai subscriber come array NumPy."""
        # convert joystick data into a numpy array
        action = np.zeros(4) 
        action[0] = self.offset_data.x
        action[1] = self.offset_data.y
        action[2] = self.offset_data.z
        action[3] = self.gripper_command.data
        
        self.last_action = action

        # self.get_logger().info(f"Action: {action}, Identical Count: {self.identical_action_count}")
        return action
    
    def reset_cmd(self):
        """funzione per resettare lo stato del nodo RecorderNode internamente."""

        self.gripper_command.data = 0.0  # Reset gripper command
        self.offset_data = Vector3()  # Resetta offsets
        self.last_action = np.zeros(4)  # Resetta last action
        self.identical_action_count = 0  # Reset counter