from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gym
import mujoco
import numpy as np
import rclpy


@dataclass(frozen=True)
class GymRenderingSpec:   ######### rimane questa? posso/devo fare un unico UR_pick_gym_env.py senza questo mujocoGym?
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

        # self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix()) ## USELESS IN OUR CASE, --> serve per parte di controllo che noi gestiamo lato ROS        
        # self._model.vis.global_.offwidth = render_spec.width
        # self._model.vis.global_.offheight = render_spec.height

        self._data = mujoco.MjData(self._model) ##############################
        ## questo è quello che devo riempre tramite ros col mio topic /MujocoState, solo quelle cose?
        ## poi in envSPecifico, quando chiama il _data.reset devo chiamare il servizio che ho creato per inviare il reset a mujoco, hiusto?
        ## ma devo fare stessa cosa anche per mj_data.step o fa in automatico? --> in teoria fa in automatico lato ROS (aggiorna mujoco automaticamente)

        # self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None ################ cancellare
        self._render_specs = render_spec
        # self._ros_node = rclpy.create_node("base_mujoco_gym_env")


    ## sto render nel mio caso sparisce e basta? faccio già il rendere su mujoco mio
    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None: ### questa non l'abbiamo implementata, volendo si può ma non  troppo utile.. comunque direi cancello anche questa
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

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
