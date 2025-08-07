from pypylon import pylon
import numpy as np
import cv2
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, "basler_config", "my_Basler_secondTrial.pfs")
# "my_Basler_config.pfs"
# "my_Basler_secondTrial.pfs"
# "my_Basler_3Trial.pfs"

# serl_robot_infra/franka_env/camera/basler_config/my_Basler_config.pfs
class BaslerCapture:
    def __init__(self, name): #  width=640, height=480,
        self.name = name
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

        self.camera.Open()

        # Load PFS (Feature Set saved from Pylon Viewer)
        if os.path.isfile(config_path):
            print(f"Loading PFS config from: {config_path}")
            pylon.FeaturePersistence.Load(config_path, self.camera.GetNodeMap(), True)
        else:
            raise FileNotFoundError(f"PFS file not found at path: {config_path}")


        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def read(self):
        if self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                grabResult.Release()
                ####################à  RUOTARE 90° x come è messa nel attachment ################
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ####################à  RUOTARE 90° ################

                return True, img
            grabResult.Release()
        return False, None

    def close(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()
        self.camera.Close()