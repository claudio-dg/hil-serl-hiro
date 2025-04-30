from pathlib import Path
from typing import Any, Tuple, Dict
import numpy as np
from gymnasium import spaces
from ur_hiro_sim.ROS_mujoco_gym_env import MujocoGymEnv, GymRenderingSpec
import rclpy
from cv_bridge import CvBridge
import numpy as np
import cv2


import time  

#  fare:
# export PYTHONPATH=$PYTHONPATH:~/ros/catkin_ws/src/hil-serl/ur_hiro_sim

bridge = CvBridge()

def ros_image_to_numpy(ros_image):
    """Converte un messaggio ROS Image in un array NumPy."""
    # Usa CvBridge per convertire il messaggio in un'immagine OpenCV
    cv_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="rgb8")
    return np.asarray(cv_image, dtype=np.uint8)


class URPickRosEnv(MujocoGymEnv):
    """Environment specifico per il task di pick-and-place con il robot Panda."""

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = 180.0,
        render_spec: GymRenderingSpec = GymRenderingSpec(),

        image_obs: bool = True,
    ):
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec, ### 
        )
        self._action_scale = action_scale

        self.camera_id = (0, 1)
        self.image_obs = image_obs

        # Definizione degli spazi di osservazione e azione
        self.observation_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "tcp_velocity": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "gripper_command": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
            }
        )

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Dict(
                        {
                            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                            "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                            "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                        }
                    ),
                    "images": spaces.Dict(
                        {
                            "left": spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                            "right": spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )
                    
        self.action_space = spaces.Box(
            low=np.asarray([-1.0, -1.0, -1.0, -1.0]), # x y z traslation + gripper = 4
            high=np.asarray([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Esegue uno step nell'environment."""


        # implementata la logica per inviare comandi al robot tramite ROS 
        # FATTO con bridge aggiunto a StateBridge già esistente
        # e aggiornare lo stato dell'environment.        

        ##### POST INVIO CONTROLLI ROBOT,  checko le observation! 
        # TODO: (da capire questione sincronia affinchè 
        # avvenga la lettura dello stato esatto subito post azione...) ####
        super().step(action)  # Invia l'azione al robot tramite ROS

        done = False  # Determina se l'episodio è terminato
        info = {}  # Informazioni aggiuntive
    
        obs = self.compute_observation()
        reward = self._compute_reward()
        success = self._is_success()

        # Check timeout
        elapsed_time = time.time() - self._start_time
        # print(f"Elapsed time: {elapsed_time:.2f} seconds")
        time_exceeded = elapsed_time > self._time_limit

        # establish if the episode is over
        done = time_exceeded or success

        
        info = {
            "succeed": success,
            "time_exceeded": time_exceeded,
            "elapsed_time": elapsed_time,
        }

        return obs, reward, done, False, info

    def compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        # Legge lo stato corrente in modo thread-safe
        with self._state_mutex:
            current_state = self._current_state
        
        #### read info from mujoco (through Ros Topic) and reshape state msgs ####
        # ------ tcp pose ------
        tcp_position = np.array([current_state.tcp_pose.pose.position.x,
                                 current_state.tcp_pose.pose.position.y,
                                 current_state.tcp_pose.pose.position.z])
        
        tcp_orientation = np.array([current_state.tcp_pose.pose.orientation.x,
                                    current_state.tcp_pose.pose.orientation.y,
                                    current_state.tcp_pose.pose.orientation.z,
                                    current_state.tcp_pose.pose.orientation.w])

        tcp_position[2] -= 0.135  # Offset lungo l'asse Z per il reale TCP (teoria 0.145)
        
        tcp_pose = np.concatenate([tcp_position,tcp_orientation]).astype(np.float32)

        obs["state"]["tcp_pose"] = tcp_pose


        # ------ tcp Vel ------

        tcp_lin_velocity = np.array([current_state.tcp_velocity.twist.linear.x,
                                    current_state.tcp_velocity.twist.linear.y,
                                    current_state.tcp_velocity.twist.linear.z])
        
        tcp_ang_velocity = np.array([current_state.tcp_velocity.twist.angular.x,
                                     current_state.tcp_velocity.twist.angular.y,
                                     current_state.tcp_velocity.twist.angular.z])
 
        tcp_velocity = np.concatenate([tcp_lin_velocity,tcp_ang_velocity]).astype(np.float32)

        obs["state"]["tcp_vel"] = tcp_velocity


        # ------ tcp force/torque  (utile nel caso in cui debba basare reward su forza, es: plug in hole task direi) ------

        tcp_force = np.array([current_state.tcp_force_torque.wrench.force.x,
                              current_state.tcp_force_torque.wrench.force.y,
                              current_state.tcp_force_torque.wrench.force.z])
        
        tcp_torque = np.array([current_state.tcp_force_torque.wrench.torque.x,
                               current_state.tcp_force_torque.wrench.torque.y,
                               current_state.tcp_force_torque.wrench.torque.z])
        
        tcp_force_torque = np.concatenate([tcp_force, tcp_torque]).astype(np.float32)

        obs["state"]["tcp_ft"] = tcp_force_torque

        # ------ gripper pose ------

        # divido per avere tra 0,1 (converto eventualmente in booleano? vediamo per ora lascio così)
        gripper_pose = np.array(current_state.gripper_command.data / 239, dtype=np.float32) # max = 239 da joystick
        obs["state"]["gripper_pose"] = gripper_pose

        # ------ images ------
        # hard code the known number of cameras 
        # if self.image_obs: ###############commenta per togliere camera
        #     obs["images"] = {} ###############commenta per togliere camera
        #     obs["images"]["right"] = ros_image_to_numpy(current_state.camera_images[0]) ###############commenta per togliere camera
           
            # obs["images"]["left"] = ros_image_to_numpy(current_state.camera_images[1])

            # DEBUG TEST
            # right_image = ros_image_to_numpy(current_state.camera_images[0])
            # print(f" *-*-*- Right image shape: {right_image.shape}, dtype: {right_image.dtype}")
            # print(right_image) # stampa valori pixels

            # cv2.imshow("Right Camera", obs["images"]["right"])
            # # cv2.imshow("Left Camera", obs["images"]["left"])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


        # ------ object pose : this case -> 1 single obj ------

        object_position = np.array([current_state.obj_poses[0].pose.position.x,
                                    current_state.obj_poses[0].pose.position.y,
                                    current_state.obj_poses[0].pose.position.z])
        
        object_orientation = np.array([current_state.obj_poses[0].pose.orientation.x,
                                       current_state.obj_poses[0].pose.orientation.y,
                                       current_state.obj_poses[0].pose.orientation.z,
                                       current_state.obj_poses[0].pose.orientation.w])
        
        object_pose = np.concatenate([object_position,object_orientation]).astype(np.float32)

        obs["state"]["pickup_object_pose"] = object_pose

        # ------ object velocity : this case -> 1 single obj------

        object_lin_velocity = np.array([current_state.obj_velocities[0].twist.linear.x,
                                    current_state.obj_velocities[0].twist.linear.y,
                                    current_state.obj_velocities[0].twist.linear.z])
        
        object_ang_velocity = np.array([current_state.obj_velocities[0].twist.angular.x,
                                       current_state.obj_velocities[0].twist.angular.y,
                                       current_state.obj_velocities[0].twist.angular.z])
        
        object_velocity = np.concatenate([object_lin_velocity,object_ang_velocity]).astype(np.float32)

        obs["state"]["pickup_object_vel"] = object_velocity

        #  Overall debug print
        # self._ros_node.get_logger().info(f" ***** Final observation: {obs}")
        return obs

    def _compute_reward(self) -> float:

        ##########################################################################
        ##########################################################################

        # The formula `np.exp(-20 * dist)` applies a steep exponential decay to the distance.
        #  When `dist` is small (close to zero), the value of `r_close` will be close to 1, as `np.exp(0)` equals 1.
        #  As `dist` increases, the value of `r_close` rapidly approaches 0 due to the negative exponent. 
        # 
        # The factor `-20` controls the rate of decay, making the function highly sensitive to changes in `dist`.
        #  Larger values for this factor result in a steeper drop-off.
        # This type of function is commonly used in robotics and reinforcement learning to reward proximity or penalize distance. For example, in a robotic manipulation task, 
        # `r_close` might represent a reward signal that encourages the robot to minimize the distance to a target object. 
        # The steep decay ensures that the reward is significant only when the robot is very close to the object, promoting precise movements.
        
        ##########################################################################
        ##########################################################################

         # Legge lo stato corrente in modo thread-safe
        with self._state_mutex:
            current_state = self._current_state

        obj_Z_init = 0.3 # hard coddo, altrimenti dovrei leggere pose iniziale nel reset, ma avendo reset nel madre è inutilmente complesso.. hard coddo e basta
        obj_Z_desired = obj_Z_init + 0.2 # hard coddo altezza desiderata = incremento di  0,2
        object_position = np.array([current_state.obj_poses[0].pose.position.x,
                                    current_state.obj_poses[0].pose.position.y,
                                    current_state.obj_poses[0].pose.position.z])
               
        tcp_position = np.array([current_state.tcp_pose.pose.position.x,
                                 current_state.tcp_pose.pose.position.y,
                                 current_state.tcp_pose.pose.position.z])
        
        tcp_position[2] -= 0.135  # Offset lungo l'asse Z per il reale TCP

        
        dist = np.linalg.norm(object_position - tcp_position)
        # print(f" ##### Distance between object and tcp: {dist}")

        decay_rate = 20 # larger values --> higher sensitivity to dist changes (rew aumenta + velocemente quando fa variazioni piccole vicine al goal finale,
        # e più lentamente negli spostamenti ampi/lontani iniziali)
        target_distance = 0.0075 #0.141 # distanza minima desiderata alla quale raggiungere rew = 1
        closeness_reward = np.exp(-decay_rate * (dist - target_distance))
        closeness_reward = np.clip(closeness_reward, 0.0, 1.0) # cappo il valore tra 0-1 per evitare che distanze inferiori a0.15 diano rew>1

        lift_reward = (object_position[2]- obj_Z_init) / (obj_Z_desired - obj_Z_init)
        lift_reward = np.clip(lift_reward, 0.0, 1.0) # cappa il valore tra 0-1

        total_reward = 0.3 * closeness_reward + 0.7 * lift_reward
        # print(f" \n\n TOT Reward: {total_reward},  \n Closeness REW: {closeness_reward},  \n Lift REW: {lift_reward}")
        return total_reward
        

    
    def _is_success(self) -> bool: #### qui c'è l'algoritmo che determina nella simulazione l'esito success

        # Legge lo stato corrente in modo thread-safe
        with self._state_mutex:
            current_state = self._current_state

        obj_Z_init = 0.3 # hard coddo, altrimenti dovrei leggere pose iniziale nel reset, ma avendo reset nel madre è inutilmente complesso.. hard coddo e basta
        object_position = np.array([current_state.obj_poses[0].pose.position.x,
                                    current_state.obj_poses[0].pose.position.y,
                                    current_state.obj_poses[0].pose.position.z])
               
        tcp_position = np.array([current_state.tcp_pose.pose.position.x,
                                 current_state.tcp_pose.pose.position.y,
                                 current_state.tcp_pose.pose.position.z])
        
        tcp_position[2] -= 0.135 #  Offset lungo l'asse Z per il reale TCP
        dist = np.linalg.norm(object_position - tcp_position)
        # print(f" ##### Distance between object and tcp: {dist}")
        lift = object_position[2] - obj_Z_init

        # return dist < 0.165 and lift > 0.2
        return dist < 0.05 and lift > 0.2


def main():
    print("Avvio dell'environment URPickRosEnv con ROS2...")

    # # Inizializza l'environment e il nodo ROS
    # ur_env = URPickRosEnv()

    # ur_env.reset()
    # rate = ur_env._ros_node.create_rate(10)
    # # prova_action = np.array([0.5, 0.0, 0.0, 1.0])
    # prova_action = np.array([0.0, 0.0, 0.0, 1.0])

    # for i in range(10000):
    #     # print (ur_env._current_state.gripper_command.data)
    #     # if i % 45 == 0:
    #     #     prova_action = -prova_action
    #     # if i % 250 == 0:
    #     #     ur_env.reset()
    #     # ur_env.step(prova_action)


    #     obs, reward, done, aaa, info = ur_env.step(prova_action)
    #     if done:
    #         print(f" \n### episode concluded with result: : {info['succeed']}")
    #         print(f" \n ################################ \n\n Final observation: {obs}")
    #         print(f"Final reward: {reward}")
    #         print(f"################################ \n\n")
    #         ur_env.reset()
        
    #     # ur_env._ros_node.get_logger().info(f"[UR Gym Env] Ricevuto stato MuJoCo Gripper: {ur_env._current_state.gripper_command.data}")
    #     rate.sleep()
    # ur_env.close()

if __name__ == "__main__":
    main()

