import argparse
import time
import mujoco
import mujoco.viewer
import numpy as np

from franka_sim import envs

# import joystick wrapper
from franka_env.envs.wrappers import JoystickIntervention, ControllerType

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="xbox", help="Controller type. xbox|ps5")

    args = parser.parse_args()
    controller_type = ControllerType[args.controller.upper()]

    env = envs.PandaPickCubeGymEnv(render_mode="human", image_obs=True)
    env = JoystickIntervention(env=env, controller_type=controller_type)

    env.reset()
    m = env.model
    d = env.data
    # intervene on position control
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for i in range(100000):
            env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            viewer.sync()
