import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING
import mujoco.viewer
from pynput import keyboard

import os
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

start_key = False
def on_press(key):
    global start_key
    try:
        if str(key) == 'Key.shift':
            start_key = True
    except AttributeError:
        pass

def main(_):
    global start_key
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # print(" \n\n\n AAA \n\n\n")
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    # print(" \n\n\n BBB \n\n\n")
    # env = config.get_environment(fake_env=False, save_video=True, classifier=False)

    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    print("Press shift to start recording.\nIf your controller is not working check controller_type (default is xbox) is configured in examples/experiments/pick_cube_sim/config.py")
    print(" \n\n\n ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ \n\n\n")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # print(" \n\n\n XXX \n\n\n")
            if start_key:
                actions = np.zeros(env.action_space.sample().shape) 
                next_obs, rew, done, truncated, info = env.step(actions)
                viewer.sync()
                returns += rew
                if "intervene_action" in info:
                    actions = info["intervene_action"]
                transition = copy.deepcopy(
                    dict(
                        observations=obs,
                        actions=actions,
                        next_observations=next_obs,
                        rewards=rew,
                        masks=1.0 - done,
                        dones=done,
                        infos=info,
                    )
                )
                trajectory.append(transition)
                
                pbar.set_description(f"Return: {returns}")

                obs = next_obs
                if done:
                    if info["succeed"]:
                        for transition in trajectory:
                            transitions.append(copy.deepcopy(transition))
                        success_count += 1
                        print(f"Success count: {success_count}")
                        pbar.update(1)
                    trajectory = []
                    returns = 0
                    obs, info = env.reset()
            if success_count >= success_needed:
                print("aaaaaaaaaaaaaaaaaa")

                break
            
        print("bbbbbbbbb")

    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

def new_func():
    return False

if __name__ == "__main__":
    app.run(main)