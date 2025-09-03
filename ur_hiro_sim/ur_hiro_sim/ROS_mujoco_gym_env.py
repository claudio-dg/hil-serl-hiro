from dataclasses import dataclass
from typing import Any, Literal, Tuple, Dict
import gym
import numpy as np
import rclpy
from hil_serl_hiro_utils.msg import SimulationState  # Custom msg
import threading
from rclpy.wait_for_message import wait_for_message
from std_srvs.srv import Trigger
from std_msgs.msg import Float32MultiArray 
import time
# from wait_for_message import wait_for_messages

@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 128 # 240
    width: int = 128 # 320
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"

class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface, ROS state is injected externally."""

    def __init__(
            self,
            seed: int = 0,
            control_dt: float = 0.02,
            physics_dt: float = 0.002,
            time_limit: float = float("inf"),
            render_spec: GymRenderingSpec = GymRenderingSpec()):
        
        # Gym Parameters 
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._render_specs = render_spec
        self._current_state = SimulationState()

        self._state_mutex = threading.Lock()  # Mutex to protect current_state

        # Ros setup
        if rclpy.ok() == False:
            rclpy.init()

        self._ros_node = rclpy.create_node("ros_gym_node")

        # Publishers
        # Publish gym's robot action on ROS
        self._action_publisher = self._ros_node.create_publisher(
            Float32MultiArray, 
            "gym_ros/robot_action", 
            1  # QoS
        )

        # Subscribers
        # Read mujoco state from ROS
        self._ros_node.create_subscription(
            SimulationState,  
            '/mujoco_state',
            self.mujoco_state_cb,
            rclpy.qos.qos_profile_sensor_data
        )

        # Services
        # Client for mujoco_reset_scene Service
        self._reset_client = self._ros_node.create_client(Trigger, "mujoco_reset_scene")

        # Wait for the service to become available
        while not self._reset_client.wait_for_service(timeout_sec=1.0):
            self._ros_node.get_logger().info("Aspettando che il servizio 'mujoco_reset_scene' sia disponibile...")

        # Spinning the node in a different thread
        self._executor = rclpy.executors.MultiThreadedExecutor()
        self._executor.add_node(self._ros_node)
        self._spinning_thread = threading.Thread(target=self.spinning_cb)
        self._spinning_thread.start()

    def __del__(self):
        """Destructor to clean up the ROS node."""
        self._ros_node.destroy_node()
        rclpy.shutdown()
        self._spinning_thread.join(timeout=1.0) # Investigate how to better handle threads

    def control_dt(self) -> float:
        return self._control_dt

    @property
    def current_state(self):
        """Ritorna lo stato aggiornato dell'environment."""
        with self._state_mutex:
            return self._current_state


    # ------------------- ROS Callbacks ------------------- #
    def mujoco_state_cb(self, msg):
        """Callback per aggiornare l'environment con i dati ricevuti da ROS."""
        with self._state_mutex:
            self._current_state = msg
        
    def publish_action(self, action: np.ndarray):
        """Pubblica l'azione sul topic ROS."""
        msg = Float32MultiArray()
        msg.data = action.tolist()  
        self._action_publisher.publish(msg)

    def spinning_cb(self):
        while rclpy.ok():
            self._ros_node.get_logger().info("Started spinning base gym node")
            # rclpy.spin(self._ros_node)
            self._executor.spin()
        self._ros_node.get_logger().info("Finished spinning base gym node")

    # ----------------------------------------------------- #

    # ------------------- Gym Overrides ------------------- #
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.publish_action(action)


    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        
        self._ros_node.get_logger().info(f" \n # Resetting MuJoCo environment")
        # call the reset service
        request = Trigger.Request()
        response = self._reset_client.call(request)    
        # response = self._reset_client.call(request, timeout_sec=25.0)    

        # wait for the result
        self._ros_node.get_logger().info(f" ####################### Reset completes with return=: {response.message}")

        # Waiting for new state from Mujoco through ROS2 msg
        self._current_state = None

        success, msg = wait_for_message(SimulationState, self._ros_node, '/mujoco_state', time_to_wait=25.0)
        # success, msg = wait_for_message(SimulationState, self._ros_node, '/mujoco_state', time_to_wait=25.0)

        if success:
            self._ros_node.get_logger().info("\n\n\n Nuovo stato ricevuto! \n\n\n ")
            self._current_state = msg
        else:
            self._ros_node.get_logger().error("Timeout: nessun messaggio ricevuto dal topic /mujoco_state.")
        
        # Init timer for the episode's timeout
        self._start_time = time.time()

    def render(self):
        # This should NOT be overritten by son
        pass


    def close(self) -> None:

        self._ros_node.get_logger().info(f"\n #Closing MuJoCo environment")
        rclpy.shutdown()
        if self._spinning_thread.is_alive():
            self._spinning_thread.join(timeout=1.0)
    # ----------------------------------------------------- #