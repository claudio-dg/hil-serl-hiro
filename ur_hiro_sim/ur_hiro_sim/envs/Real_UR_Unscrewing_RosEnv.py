from pathlib import Path
from typing import Any, Tuple, Dict
import numpy as np
from gymnasium import spaces
# from ur_hiro_sim.ROS_mujoco_gym_env import MujocoGymEnv 
from ur_hiro_sim.Real_ROS_gym_env import RealGymEnv, GymRenderingSpec
import rclpy
from cv_bridge import CvBridge
import numpy as np
import cv2
import time  
from franka_env.camera.video_capture import VideoCapture
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.my_Basler_capture import BaslerCapture
from franka_env.utils.rotations import euler_2_quat, quat_2_euler
import cv2
import copy
from gymnasium import spaces
import queue
import threading
import os
import time
from datetime import datetime
from collections import OrderedDict

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def print_blue(x):
    return print("\033[95m {}\033[00m".format(x))

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True 
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                # [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k], axis=1
                # [cv2.resize(v, (512, 512)) for k, v in img_array.items() if "full" not in k], axis=1
                [cv2.resize(v, (650, 650)) for k, v in img_array.items() if "full" not in k], axis=1 #solo grandezza finestre cv, non cambia risoluzione immagini mostrate a schermo
                #  per risoluz immagini, OTLRE A QUA, va vambiato gym renderin spec in "Real_ROS_gym_env.py"

            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)

bridge = CvBridge()
class Real_UR_Unscrewing_RosEnv(RealGymEnv):
    """Environment specifico per il task di pick-and-place con il robot UR."""

    def __init__(
        self,
        action_scale: np.ndarray = np.asarray([0.1, 1]),
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        time_limit: float = 50.0, 
        render_spec: GymRenderingSpec = GymRenderingSpec(),

        image_obs: bool = True,

        start_camera: bool = True, 
    ):
        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=time_limit,
            render_spec=render_spec,  
        )
        self._action_scale = action_scale
        self.camera_id = (0, 1)
        self.image_obs = image_obs
        ################################################################################################################
        self.display_image = True
        self.url: str = "http://127.0.0.2:5000/"

        self.REALSENSE_CAMERAS: Dict = {
        "my_realsense": {
            "serial_number": "130322273284",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        }

        # print_green(f" \n\n\n\n\n\n START CAMERA flag =  {start_camera} \n\n\n\n\n\n  ")

        # definisce un dizionario chiamato IMAGE_CROP che associa a "my_realsense" una funzione lambda 
        # (cioè una funzione anonima) 
        # che prende in input un’immagine img e restituisce una sotto-porzione (crop) di essa.
        # self.IMAGE_CROP: dict[str, callable] = {"my_realsense": lambda img: img[50:-200, 200:-200]} 
        #50:-200 sulle righe (asse Y):
        # Prende le righe dalla 50-esima fino a 200 righe dalla fine.
        # 200:-200 sulle colonne (asse X):
        # Prende le colonne dalla 200-esima fino a 200 colonne dalla fine.


        # self.IMAGE_CROP: dict[str, callable] = {"my_realsense": lambda img: img[200:-200, 350:-300]} # provo crop più dettagliato sul bit
        # sembra ok, provare ahhiungere basler crop -->
        # self.IMAGE_CROP: dict[str, callable] = {
        # "my_realsense": lambda img: img[300:-300, 450:-350], #### 1280x720 (x quadrata: (1280-720):2 = 560:2 = 280.... quindi 280    )
        # "my_basler": lambda img: img[50:-150, 100:-50],  # <-- sDA VALUTARE BENE QUA E TESTARE
        # }

        #  QUADRATA OER CROPPARE 720x720 --> img[:720, 280:1000] RSENSE   lambda img: img[280:1000, :720 ] BASLER
        #  PER CROPPARE 520 x 520 --> lambda img: img[100:620, 380:900]  RSENSE((?)) (QUINDI IMPORTANTE TENERE RATEO 620-100 = 520 === 900-380 = 520 !!!!!)
        #  per decentrare crop quindi ad esempio va bene anche 910-390=520.. FARLO E ANCHE IN DART
        # diminuisco ancora crop, es: lo voglio 220 ---->img[100:620, 380:900] diventa: img[250:470, 530:750]

        self.IMAGE_CROP: dict[str, callable] = {
        "my_realsense": lambda img: img[250:470, 570:790], #### 1280x720 (x quadrata: (1280-720):2 = 560:2 = 280.... quindi 280 di offsett su y   )
        # per croppare di quadrato maggiore aggiungo stesso offset (oltre a 280) sia x che y es: altri 200(x2) --> [380x & 100y]
        "my_basler": lambda img: img[350:930, 170:750 ],  # <-- sDA VALUTARE BENE QUA E TESTARE
        }

        self.save_video = False #True # False
        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        ################################################################################################################

        # Definizione degli spazi di osservazione e azione
        self.observation_space = spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                # "tcp_velocity": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                'tcp_ft':  spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
            }
        )

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Dict(
                        {
                            "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                            # "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32), # potrebbe servire anche velocità per vedere che sia fermo? mh, in caso devo aggiungere publisher di vel come ho fatto su asus
                            'tcp_ft':  spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                            "gripper_pose": spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
                        }
                    ),
                    "images": spaces.Dict(
                        {
                            "my_realsense": spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                            "my_basler": spaces.Box(
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
            low=np.asarray([-1.0, -1.0, -1.0, -1.0]), # x y z traslation + booleano avvitatore (o floar se si unaput analogicfo + complesso...) = 4 (eventualmente ROT qua?)
            high=np.asarray([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        ####################################### STARTING CAMERAS ########################################################
        if start_camera:
            self.cap = None
            self.init_cameras(self.REALSENSE_CAMERAS)
            if self.display_image:
                self.img_queue = queue.Queue()
                self.displayer = ImageDisplayer(self.img_queue, self.url)
                self.displayer.start()

    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:  # close cameras if they are already open
            self.close_cameras()

        self.cap = OrderedDict()
        # Read "all" Realsense cameras
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

        # Add Basler camera Capture
        self.cap["my_basler"] = VideoCapture(BaslerCapture(name="basler1"))


    def close_cameras(self):
        """Close both wrist cameras."""
        try:
            for cap in self.cap.values():
                cap.close()
        except Exception as e:
            print(f"Failed to close cameras: {e}")


    def get_im(self) -> Dict[str, np.ndarray]:
        """Get images from the realsense cameras & my Basler."""
        images = {}
        display_images = {}
        full_res_images = {}  # New dictionary to store full resolution cropped images
        for key, cap in self.cap.items():
            try:
                rgb = cap.read()
                cropped_rgb = self.IMAGE_CROP[key](rgb) if key in self.IMAGE_CROP else rgb 
                resized = cv2.resize(
                    cropped_rgb, self.observation_space["images"][key].shape[:2][::-1]
                )
                
                images[key] = resized[..., ::-1]# provo a non resizare quelle datein pasto a rete
                # images[key] = rgb
                
                display_images[key] = resized
                display_images[key + "_full"] = cropped_rgb

                full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
                # full_res_images[key] = copy.deepcopy(rgb)  # Store the full resolution NON cropped image # IO
            except queue.Empty:
                input(
                    f"{key} camera frozen. Check connect, then press enter to relaunch..."
                )
                cap.close()
                self.init_cameras(self.REALSENSE_CAMERAS)
                return self.get_im()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image:
            # self.img_queue.put(display_images) ######################################## provo a commentare
            self.img_queue.put(full_res_images) ######################################## e mostrare fully res images
        return images
    
    def save_video_recording(self):
        try:
            if len(self.recording_frames):
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                
                for camera_key in self.recording_frames[0].keys():
                    if self.url == "http://127.0.0.1:5000/":
                        video_path = f'./videos/left_{camera_key}_{timestamp}.mp4'
                    else:
                        video_path = f'./videos/my_right_{camera_key}_{timestamp}.mp4'
                    
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]
                    
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )
                    
                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])
                    
                    video_writer.release()
                    print_green(f"Saved video for camera {camera_key} at {video_path}")
                
            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Esegue uno step nell'environment."""

        super().step(action)  # Send action to robot through ROS
        # print_blue(f"[DEBUG] AZIONE POLICY finaleee GYM SPECIFICO: {action}")

        info = {}
        obs = self.compute_observation()
        ###################### caso REC DEMO REALE ######################
        reward = self._compute_reward()
        # success, is_low_enough = self._is_success()
        success, is_inserted_enough = self._is_success()

        # Check timeout
        elapsed_time = time.time() - self._start_time
        time_exceeded = elapsed_time > self._time_limit

        # establish if the episode is over
        done = time_exceeded or success 

        info = {
            # "succeed": success,
            "time_exceeded": time_exceeded,
            "elapsed_time": elapsed_time,
            #### add info to establish real case's success along with image classifier
            "is_inserted_enough": is_inserted_enough, # qui ci posso mettere un controllo sulla forza misurata
        }

        return obs, reward, done, False, info

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset dell'environment."""
        super().reset(seed=seed, **kwargs)
        obs = self.compute_observation()
        # save video ?
        if self.save_video:
            self.save_video_recording()

        return obs, {}
    
    def compute_observation(self) -> dict:
        """ read info from Real robot (through Ros Topic) and reshape state msgs to fill obs """  

        obs = {}
        obs["state"] = {}

        # Read current_state in a thread-safe way
        with self._state_mutex:
            current_state = self._current_state
        

        # ------ tcp pose ------

        tcp_position = np.array([current_state.tcp_pose.pose.position.x,
                                 current_state.tcp_pose.pose.position.y,
                                 current_state.tcp_pose.pose.position.z])
        
        tcp_orientation = np.array([current_state.tcp_pose.pose.orientation.x,
                                    current_state.tcp_pose.pose.orientation.y,
                                    current_state.tcp_pose.pose.orientation.z,
                                    current_state.tcp_pose.pose.orientation.w])

        tcp_pose = np.concatenate([tcp_position,tcp_orientation]).astype(np.float32)

        obs["state"]["tcp_pose"] = tcp_pose


        # ------ tcp Vel ------

        # tcp_lin_velocity = np.array([current_state.tcp_velocity.twist.linear.x,
        #                             current_state.tcp_velocity.twist.linear.y,
        #                             current_state.tcp_velocity.twist.linear.z])
        
        # tcp_ang_velocity = np.array([current_state.tcp_velocity.twist.angular.x,
        #                              current_state.tcp_velocity.twist.angular.y,
        #                              current_state.tcp_velocity.twist.angular.z])
 
        # tcp_velocity = np.concatenate([tcp_lin_velocity,tcp_ang_velocity]).astype(np.float32)

        # obs["state"]["tcp_vel"] = tcp_velocity


        # ------ tcp force/torque  (utile nel caso in cui debba basare reward su forza, es: peg in hole task direi) ------

        tcp_force = np.array([current_state.tcp_force_torque.wrench.force.x,
                              current_state.tcp_force_torque.wrench.force.y,
                              current_state.tcp_force_torque.wrench.force.z])
        
        tcp_torque = np.array([current_state.tcp_force_torque.wrench.torque.x,
                               current_state.tcp_force_torque.wrench.torque.y,
                               current_state.tcp_force_torque.wrench.torque.z])
        
        tcp_force_torque = np.concatenate([tcp_force, tcp_torque]).astype(np.float32)

        obs["state"]["tcp_ft"] = tcp_force_torque

        # print_green(f" *** FORCE TORQUE: {tcp_force_torque}")
        #### quando in vite, il secondo valore (direi forza-Y) passa da valore positivo a circa -8

        # ------ gripper pose ------

        # divido per avere tra 0,1 (converto eventualmente in booleano? vediamo per ora lascio così)
        gripper_pose = np.array(current_state.gripper_state.data / 239, dtype=np.float32) # max = 239 da joystick
        obs["state"]["gripper_pose"] = gripper_pose

        # ------ images ------
        # hard code the known number of cameras 
        if self.image_obs: 
           
            real_camera_images = self.get_im()
            obs["images"] = {}
            obs["images"] = real_camera_images ## 

        # self._ros_node.get_logger().info(f" ***** Final observation: {obs}")
        return obs

    def _compute_reward(self) -> float:

        # Read current_state in a thread-safe way
        with self._state_mutex:
            current_state = self._current_state

        tcp_Z_init = 0.43 # hard code
        tcp_Z_desired = tcp_Z_init - 0.235 # hard code desired height = increasing of 0,2
               
        tcp_position = np.array([current_state.tcp_pose.pose.position.x,
                                 current_state.tcp_pose.pose.position.y,
                                 current_state.tcp_pose.pose.position.z])
        
        height_reward = (tcp_position[2]- tcp_Z_init) / (tcp_Z_desired - tcp_Z_init)
        height_reward = np.clip(height_reward, 0.0, 1.0) # cap between 0-1

        return height_reward
        

    
    def _is_success(self) -> bool: 
        
        #  qui devo mettere classificatore come hilserl, vedere.. per ora metto posa robot
        # Legge lo stato corrente in modo thread-safe
        with self._state_mutex:
            current_state = self._current_state

        # obj_Z_init = 0.3# hard code

        # object_position = np.array([current_state.obj_poses[0].pose.position.x,
        #                             current_state.obj_poses[0].pose.position.y,
        #                             current_state.obj_poses[0].pose.position.z])
               
        tcp_position = np.array([current_state.tcp_pose.pose.position.x,
                                 current_state.tcp_pose.pose.position.y,
                                 current_state.tcp_pose.pose.position.z])
        # inutile ma per ora lascio pos
        
        tcp_force = np.array([current_state.tcp_force_torque.wrench.force.x,
                              current_state.tcp_force_torque.wrench.force.y,
                              current_state.tcp_force_torque.wrench.force.z])
        
        is_inserted_enough = tcp_force[1] < -1.0 # force.y
        # tcp_position[2] -= 0.135 #  Offset lungo l'asse Z per il reale TCP
        # dist = np.linalg.norm(object_position - tcp_position)
        # print(f" ##### Distance between object and tcp: {dist}")
        # lift = object_position[2] - obj_Z_init

        # return "success" (T/F), "is_low_enough (T/F)"
        # return tcp_position[2]  < 0.2, tcp_position[2]  < 0.26 # LOW ENOUGH
        return tcp_position[2]  < 0.2, is_inserted_enough # HIGH ENOUGH (gripper case)

def main():
    print("Avvio dell'environment Real_UR_Unscrewing_RosEnv con ROS2...")

if __name__ == "__main__":
    main()

