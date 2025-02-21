"""Simple example showing how to get gamepad events."""

from __future__ import print_function
from inputs import get_gamepad
####################
from dataclasses import dataclass
from enum import Enum
import inputs
import threading
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class ControllerType(Enum):
    XBOX = "xbox"

@dataclass
class ControllerConfig:
    resolution: dict
    scale: dict

class JoystickIntervention():
    CONTROLLER_CONFIGS = {
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
    def __init__(self,  action_indices=None, controller_type=ControllerType.XBOX):
        
        self.gripper_enabled = True
        # if self.action_space.shape == (6,):
        #     self.gripper_enabled = False
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
                        # if self.controller_type == ControllerType.PS5:
                        #     normalized_value = (current_value - (resolution / 2)) / (resolution / 2)
                        # else:
                        normalized_value = current_value / (resolution / 2) # fa diviso Res/2 perchè poi scala in modo da avere da  -TOT/2 a +TOT/2 anziche da 0 a TOT (una roba così)
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
                            #print("777 cmd -- ABS HATOX", code, self.rz_axis) # nscoperto dovrebbe essere freccia sx però in BTN printa code = HATOX.. quindi c'è qualche imprecisione di codice ma hatox è RB o LB
                        
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

            #provo a mettere qua gestione/filtro dei comandi
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
            # print("INTERVENUTO VAR = ", intervened)
            if np.linalg.norm(expert_a) > 0.001 or self.left or self.right:
                intervened = True

            print("------- INTERVENUTO VAR = ", intervened)
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
                # return expert_a, True
                ## CONCETTUALMENTE POTREI FARE PROVA METTENDO QUA IL PUBLISHER in questo IF
    
    ##############################################################################
    ##############################################################################
    # QUINDI RISULTATO QUI AL MOMENTO HO:
    # CODICE CHE LEGGE INPUT DEL JOYSTICK, CONTROLLA CHE SIANO OLTRE UNA SOGLIA DI DEADZONE, E CREA UN ARRAY DI 7 VALORI CHIAMATO expert_a[]
    # DOVRÒ QUINDI ESTRARRE PRIMI 3 VALORI --> i.e. delta lungo xyz 
    # --> valutare se necessitano ulteriori modifiche (es moltiplic per 0.01 come fa lui?) 
    # --> e aggiungerli alla posa corrente da pubblicare sul topic robot!
    # ESTRARRE poi settimo valore [6] contenente info gripper
    ##############################################################################
    ##############################################################################

    ##############################################################################
    ##############################################################################
    DEVO PERO' FARE UN NODO ROS2 PER FARE SUB PUB ECC E METTERLO DENTRO UN PKG ROS.. MENTRE ORA È SOLO UNO SCRIPT PYTHON... TODO
    ##############################################################################
    ##############################################################################

        
def main():
    """Just print out some event infomation when the gamepad is used."""
    # while 1:
    #     events = get_gamepad()
    #     for event in events:
    #         print(event.ev_type, event.code, event.state)
    my_gamepad = JoystickIntervention()
    print("Controller thread started. Listening for inputs...")
    
    try:
        while True:
            time.sleep(1)  # Mantieni attivo il programma
    except KeyboardInterrupt:
        print("Exiting program...")
        my_gamepad.running = False
        my_gamepad.thread.join()  # Aspetta la terminazione del thread



if __name__ == "__main__":
    main()
