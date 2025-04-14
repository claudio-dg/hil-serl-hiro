from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import threading

import gym
import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger  # Importa il servizio Trigger

# Inizializza il sistema ROS
rclpy.init()

@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 128
    width: int = 128
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        render_spec: GymRenderingSpec = GymRenderingSpec(),
    ):
        print(f"xml_path: {xml_path}")  # Aggiungi questa linea per debug

        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        # questo qui non mi interessa --> le cose di mujoco le faccio in CPP (mujoco_ros2_control_node)
        # e devo poi farle leggere qui o da qualche parte in python tramite topic ROS

        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height

        self._data = mujoco.MjData(self._model)
        # questo  come sopra, preso in CPP tramite mj_MakeData

        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec
        self._ros_node = rclpy.create_node("base_mujoco_gym_env")

        # Crea un client per il servizio /close_render_service
        self._close_render_client = self._ros_node.create_client(Trigger, '/close_render_service')

        # Start the ROS spinning in a separate thread
        self._ros_spin_thread = threading.Thread(target=self.spin)
        self._ros_spin_thread.start()

    def spin(self): # sbagliat0...
        rclpy.spin(self._ros_node)

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

        # Chiama il servizio /close_render_service
        if self._close_render_client.wait_for_service(timeout_sec=1.0):
            request = Trigger.Request()
            future = self._close_render_client.call_async(request)
            
            # Attendi il completamento del futuro senza bloccare il thread principale
            while not future.done():
                rclpy.spin_once(self._ros_node, timeout_sec=0.1)
            
            if future.result() is not None:
                print(f"Service call succeeded: {future.result().message}")
            else:
                print(f"Service call failed: {future.exception()}")
        else:
            print("Service /close_render_service not available")

        # Chiama il metodo per chiudere il nodo ROS
        self.shutdown_ros()

    def shutdown_ros(self) -> None:
        # Chiudi il nodo ROS
        self._ros_node.destroy_node()
        rclpy.shutdown()
        # Join the ROS spin thread
        self._ros_spin_thread.join()

    def time_limit_exceeded(self) -> bool:
        return self._data.time >= self._time_limit

    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random



if __name__ == "__main__":
     # Crea un'istanza della classe MujocoGymEnv
    prova = MujocoGymEnv(xml_path=Path('/home/claudiodelgaizo/ros/catkin_ws/src/hil-serl/franka_sim/franka_sim/envs/xmls/arena.xml'))

    prova.close()