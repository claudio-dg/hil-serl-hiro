#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from sensor_msgs.msg import Image
from hil_serl_hiro_utils.msg import RealState  # Custom message
from std_msgs.msg import Float32MultiArray

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

##########
import numpy as np
import gymnasium as gym
from std_srvs.srv import Trigger 
from ur_msgs.msg import IOStates
import ur_msgs.srv
import threading
import time
##########

##################################################################
import tf_transformations
import random

# def quat_to_euler(q):
    # return tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])

def euler_to_quat(roll, pitch, yaw):
    q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
    return q  # [x, y, z, w]
##################################################################

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))
 
def print_orange(x):
    return print("\033[93m {}\033[00m".format(x))
 
def print_blu(x):
    return print("\033[94m {}\033[00m".format(x))

class RealStateBridgeNode(Node):
    def __init__(self):
        super().__init__('real_state_bridge_node')

        ## z: z: 0.32363064023408317 --> H vite circa, un po meno in realtà tipo 0,30840358685060526
        self.switch = -1
        first_POSITION = np.array([-0.5031390929532893, 0.14744824937867473, 0.36])
        seco_POSITION = np.array([-0.5031390929532893, 0.14744824937867473, 0.36])
        # POSE_LIMIT_HIGH =  TEMP_REAL_POSITION + np.array([0.035, 0.035, 0.09]) # leggermente più ampio ma cappo azioni lato wrapper, per avere più libertà MA movimenti più precisi 
        # POSE_LIMIT_LOW = TEMP_REAL_POSITION - np.array([0.035, 0.035, 0.0701])
         
        first_POSITION = np.array([-0.5031390929532893, 0.14744824937867473, 0.36])
        second_POSITION = np.array([-0.5031390929532893, 0.14744824937867473, 0.36])

        # safety boundary box
        # self.xyz_bounding_box = gym.spaces.Box(
        #     POSE_LIMIT_LOW[:3],
        #     POSE_LIMIT_HIGH[:3],
        #     dtype=np.float64,
        # )


        self.tcp_pose = PoseStamped()
        # self.tcp_velocity = TwistStamped()
        self.tcp_force_torque = WrenchStamped()

        self.robot_action = Float32MultiArray()  # Robot action given by Gym

        # Configurazione QoS per il publisher e il subscriber di robot action, per evitare code lunghe non-gestite ma prendere sempre ultimo msg (in teoria)
        qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,  # Garantisce la consegna dei messaggi
        durability=DurabilityPolicy.VOLATILE,   # I messaggi non vengono conservati
        depth=1                                 # Mantiene solo l'ultimo messaggio nella coda
)

        # --- Robot Topics Subscribers
        # self.create_subscription(PoseStamped, '/admittance_controller/w_T_ee', self.tcp_pose_callback, qos_profile)
        # self.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self.tcp_force_torque_callback, 10)

        # -- Robot's Controller Goal Publisher
        self.tcp_goal_publisher = self.create_publisher(PoseStamped, 'admittance_controller/target_pose', qos_profile)

    # Function to clip robot actions within safety boundaries
    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """Clip the pose to be within the safety box."""
        pose[:3] = np.clip(
            pose[:3], self.xyz_bounding_box.low, self.xyz_bounding_box.high
        ) 
        return pose

    def move_robot(self):

        new_action_pose = PoseStamped()
        new_action_pose.header.stamp = self.get_clock().now().to_msg()
        new_action_pose.header.frame_id = 'base_link'

        new_action_pose.pose.position.x = -0.5 #+ translation.data[0]
        new_action_pose.pose.position.y = 0.14 + (0.15*self.switch)
        new_action_pose.pose.position.z = 0.45 #+  translation.data[2]
        new_action_pose.pose.orientation.x = -0.5061135480338557
        new_action_pose.pose.orientation.y = -0.4931089242003056
        new_action_pose.pose.orientation.z = 0.49645543168566075
        new_action_pose.pose.orientation.w = 0.5042069711144461


        print_orange(f"\n  goal originale: {new_action_pose.pose.position.x}, {new_action_pose.pose.position.y}, {new_action_pose.pose.position.z}")
 
        # # Clip robot action 
        # clipped_position = self.clip_safety_box(
        #     np.array([
        #         new_action_pose.pose.position.x,
        #         new_action_pose.pose.position.y,
        #         new_action_pose.pose.position.z
        #     ])
        # )
        
        # new_action_pose.pose.position.x = clipped_position[0]
        # new_action_pose.pose.position.y = clipped_position[1]
        # new_action_pose.pose.position.z = clipped_position[2]
        # print_green(f" posa goal CLIPPATA: {new_action_pose.pose.position.x}, {new_action_pose.pose.position.y}, {new_action_pose.pose.position.z}")
 
        self.tcp_goal_publisher.publish(new_action_pose)

        
def main(args=None):
    rclpy.init(args=args)
    node = RealStateBridgeNode()
    
    # Async Thread to avoid Race Conditions and Deadlocks
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    # Execute on separate Thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        while rclpy.ok():
            # Alterna lo switch
            node.switch *= -1 
            node.move_robot()
            time.sleep(10) 
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
