import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
from franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import requests
from scipy.spatial.transform import Rotation as R
from franka_env.envs.franka_env import FrankaEnv
from typing import List
import inputs
import threading
from dataclasses import dataclass
from enum import Enum

sigmoid = lambda x: 1 / (1 + np.exp(-x))

class HumanClassifierWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    """

    def __init__(self, env: Env, reward_classifier_func, target_hz = None):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward(self, obs):
        if self.reward_classifier_func is not None:
            return self.reward_classifier_func(obs)
        return 0
    #### questo step ad esempio direi che modifica solo i campi REWARD (rew) DONE (done) e info dell'env
    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs) ###
        done = done or rew ###
        info['succeed'] = bool(rew) ###
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
           # questo ritorno è quello che viene passato al prossimo wrapper
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
class SpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.expert = SpaceMouseExpert()
        self.left, self.right = False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        expert_a, buttons = self.expert.get_action()
        self.left, self.right = tuple(buttons)
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
            expert_a[:6] += np.random.uniform(-0.5, 0.5, size=6)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

class DualSpacemouseIntervention(gym.ActionWrapper):
    def __init__(self, env, action_indices=None, gripper_enabled=True):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.expert = SpaceMouseExpert()
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        intervened = False
        expert_a, buttons = self.expert.get_action()
        self.left1, self.left2, self.right1, self.right2 = tuple(buttons)


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty, self.penalty
        else:
            return reward, 0.0

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward, penalty = self.reward(reward, action)
        info["grasp_penalty"] = penalty
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info




class ControllerType(Enum):
    PS5 = "ps5"
    XBOX = "xbox"

@dataclass
class ControllerConfig:
    resolution: dict
    scale: dict

class JoystickIntervention(gym.ActionWrapper):
    CONTROLLER_CONFIGS = {
        ControllerType.PS5: ControllerConfig(
            # PS5 controller joystick values have 8 bit resolution [0, 255]
            resolution={
                'ABS_X': 2**8,
                'ABS_Y': 2**8,
                'ABS_RX': 2**8,
                'ABS_RY': 2**8,
                'ABS_Z': 2**8,
                'ABS_RZ': 2**8,
                'ABS_HAT0X': 1.0,
            },
            scale={
                'ABS_X': 0.4,
                'ABS_Y': 0.4,
                'ABS_RX': 0.5,
                'ABS_RY': 0.5,
                'ABS_Z': 0.8,
                'ABS_RZ': 1.2,
                'ABS_HAT0X': 0.5,
            }
        ),
        ControllerType.XBOX: ControllerConfig(
            # XBOX controller joystick values have 16 bit resolution [0, 65535]
            resolution={
                'ABS_X': 2**16,
                'ABS_Y': 2**16,
                'ABS_RX': 2**16,
                'ABS_RY': 2**16,
                'ABS_Z': 2**8,
                'ABS_RZ': 2**8,
                'ABS_HAT0X': 1.0,
            },
            scale={
                'ABS_X': -0.1,
                'ABS_Y': -0.1,
                'ABS_RX': 0.3,
                'ABS_RY': 0.3,
                'ABS_Z': 0.05,
                'ABS_RZ': 0.05,
                'ABS_HAT0X': 0.3,
            }
        ),
    }

    def __init__(self, env, action_indices=None, controller_type=ControllerType.XBOX):
        super().__init__(env)
        
        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False
        # we can set action_indices to choose which action to intervene on
        # e.g. action_indices=[0, 1, 2] will only intervene on the position control
        self.action_indices = action_indices 
        self.controller_type = controller_type
        self.controller_config = self.CONTROLLER_CONFIGS[controller_type]
        
        # Controller state
        self.x_axis = 0
        self.y_axis = 0
        self.z_axis = 0
        self.rx_axis = 0
        self.ry_axis = 0
        self.rz_axis = 0
        self.left = False   # Left bumper for close gripper
        self.right = False  # Right bumper for open gripper
        
        # Start controller reading thread
        self.running = True
        self.thread = threading.Thread(target=self._read_gamepad)
        self.thread.daemon = True
        self.thread.start()
    
    def _reset_cmds(self):
        self.x_axis = 0
        self.y_axis = 0
        self.z_axis = 0
        self.rx_axis = 0
        self.ry_axis = 0
        self.rz_axis = 0
    
    def _read_gamepad(self):
        useful_codes = ['ABS_X', 'ABS_Y', 'ABS_RX', 'ABS_RY', 'ABS_Z', 'ABS_RZ', 'ABS_HAT0X']
        
        # Store consecutive event counters and values
        event_counter = {
            'ABS_X': 0,
            'ABS_Y': 0,
            'ABS_RX': 0,
            'ABS_RY': 0,
            'ABS_Z': 0,
            'ABS_RZ': 0,
            'ABS_HAT0X': 0,
        }
    
        while self.running:
            try:
                # Get fresh events
                events = inputs.get_gamepad()
                latest_events = {}
                for event in events:
                    latest_events[event.code] = event.state
                # Process events
                for code in useful_codes:
                    if code in latest_events:
                        event_counter[code] += 1
                        current_value = latest_events[code]
                                            
                    # Only update if we've seen the same value 2 times
                    if event_counter[code] >= 1:
                        # Calculate relative changes based on the axis
                        # Normalize the joystick input values to range [-1, 1] expected by the environment
                        resolution = self.controller_config.resolution[code]
                        if self.controller_type == ControllerType.PS5:
                            normalized_value = (current_value - (resolution / 2)) / (resolution / 2)
                        else:
                            normalized_value = current_value / (resolution / 2)
                        scaled_value = normalized_value * self.controller_config.scale[code]

                        # sti nomi ABS_Y HAT0X ecc sono mezzi a caso dati dal lettore del gamepad non da sto codice -- vedi my_inpuptGamepad
                        if code == 'ABS_Y': # su giu cursore (avanti indietro robot) indietro -0.1 avanti +0.1
                            self.x_axis = scaled_value
                            #print("1 cmd -- ABS Y = L3 UP_DOWN", code, self.x_axis)
                        elif code == 'ABS_X': # dx sx cursore.. destra max = -0.1 sx max = 0.1
                            self.y_axis = scaled_value
                            #print("2 cmd -- ABS X  = L3 RIGHT_LEFT ", code,  self.y_axis)
                        elif code == 'ABS_RZ': # grilleto DX (RT) 0=non premuto max = 0.4 premuto .. fa salire robot
                            self.z_axis = scaled_value
                            #print("3 cmd -- ABS RZ = RT", code, self.z_axis)
                        elif code == 'ABS_Z': # grilleto LT max = -0.4 min 0
                            # Flip sign so this will go in the down direction
                            self.z_axis = -scaled_value
                            #print("4  cmd -- ABS Z = LT", code, self.z_axis)
                        elif code == 'ABS_RX': # cursore R3 destra sinistra .. ma non mi sembra sia mappato al robot qua.. comunque da -0.3  a 0.3
                            self.rx_axis = scaled_value
                            #print("5  cmd -- ABS RX R3", code, self.rx_axis)
                        elif code == 'ABS_RY': # da -0.3 a 0.3
                            self.ry_axis = scaled_value
                            #print("6  cmd -- ABS RY R3", code, self.ry_axis) 
                        elif code == 'ABS_HAT0X':
                            self.rz_axis = scaled_value
                            print("777 cmd -- ABS HATOX", code, self.rz_axis) # nscoperto dovrebbe essere freccia sx però in BTN printa code = HATOX.. quindi c'è qualche imprecisione di codice ma hatox è RB o LB
                        # Reset counter after update
                        event_counter[code] = 0
                        #print("AAAAAAAA_cmd", code, self.x_axis, self.y_axis, self.z_axis, self.rx_axis, self.ry_axis, self.rz_axis)
                        #print("AAAAAAAA_cmd", code, self.rz_axis)
                
                # Handle button events immediately
                if 'BTN_TL' in latest_events:
                    self.left = bool(latest_events['BTN_TL'])
                    self._reset_cmds()
                    print("\n LB --> OPEN GRIPPER", code, self.rz_axis)
                if 'BTN_TR' in latest_events:
                    self.right = bool(latest_events['BTN_TR'])
                    self._reset_cmds()
                    print("\n RB --> CLOSE GRIPPER", code, self.rz_axis)
                
            except inputs.UnpluggedError:
                print("No controller found. Retrying...")
                time.sleep(1)

    def action(self, action):
        """
        Input:
        - action: policy action
        Output:
        - action: joystick action if nonzero; else, policy action
        """
        # Get joystick action
        deadzone = 0.03 #0.03
        expert_a = np.zeros(6)
        
        expert_a[0] = self.x_axis if abs(self.x_axis) > deadzone else 0
        expert_a[1] = self.y_axis if abs(self.y_axis) > deadzone else 0
        expert_a[2] = self.z_axis if abs(self.z_axis) > deadzone else 0

        # Apply deadzone to rotation control -- ma non lin considera questi secondo me...
        expert_a[3] = self.rx_axis if abs(self.rx_axis) > deadzone else 0
        expert_a[4] = self.ry_axis if abs(self.ry_axis) > deadzone else 0
        expert_a[5] = self.rz_axis if abs(self.rz_axis) > deadzone else 0

        self._reset_cmds()
        
        intervened = False
        if np.linalg.norm(expert_a) > 0.001 or self.left or self.right:
            intervened = True


        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)
            # expert_a[:6] += np.random.uniform(-0.5, 0.5, size=6)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices] # spiega a riga  468: specifica quali dimensioni considerare e quali no
            # filtered_expert_a = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            print("[DEBUG] Expert action intervened. Values: \n X = ", expert_a[0], " Y = ", expert_a[1], " Z = ", expert_a[2], " RX = ", expert_a[3], " RY = ", expert_a[4], " Rz = ", expert_a[5], "gripper = ", expert_a[6])  # Debug print
            return expert_a, True

        #print("[ZZZ DEBUG] No intervention. Returning policy action.")  # Debug print
        return action, False

    ####### CAPITO COME FUNZIONA STEP!! ########### 
    
    # --> nel record_demos_sim.py viene chiamato il set .. esso va a richiamare ricorsivamente 
    # (dal + esterno/ultimo al più interno/primo) i metodi "step" di tutti i wrapper che sono stati usati per creare l'environment
    #ognuno di questi vari step MODIFICA una porzione dei RISULTATO (ossia delle obs,rew,done ecc) in modo da arrivare all'output finale dello step
    # al quale avrà contribuito ciascuno dei wrapper

  

    ####### CAPITO COME FUNZIONA STEP!! ########### 
    # lo step di questo wrapper ad esempio, si vede che prende IL RISULTATO dello step del wrapper precedente (ossia l'env)
    #  e ne modifica solo il campo INFO!!!!
    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        self.running = False
        self.thread.join()
        super().close()
