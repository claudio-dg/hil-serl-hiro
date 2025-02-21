import sys
import os
import cv2
import time
import queue
import copy
import numpy as np

# Aggiungi il percorso del modulo franka_env al PYTHONPATH
sys.path.append("/home/claudiodelgaizo/ros/catkin_ws/src/hil-serl/serl_robot_infra")

from camera.video_capture import VideoCapture
from camera.rs_capture import RSCapture

def crop_image(image, crop_size=(300, 300)):
    """Ritaglia l'immagine al centro con le dimensioni specificate."""
    h, w, _ = image.shape
    ch, cw = crop_size
    start_x = w // 2 - cw // 2
    start_y = h // 2 - ch // 2
    return image[start_y:start_y + ch, start_x:start_x + cw]

def get_im(caps, config, observation_space, save_video=False, display_image=False, img_queue=None, recording_frames=None):
    """Get images from the realsense cameras."""
    images = {}
    display_images = {}
    full_res_images = {}  # New dictionary to store full resolution cropped images
    for key, cap in caps.items():
        try:
            rgb = cap.read()
            cropped_rgb = config["IMAGE_CROP"][key](rgb) if key in config["IMAGE_CROP"] else rgb
            resized = cv2.resize(
                cropped_rgb, observation_space["images"][key].shape[:2][::-1]
            )
            images[key] = resized[..., ::1]  # Convert BGR to RGB -1
            display_images[key] = resized
            display_images[key + "_full"] = cropped_rgb
            full_res_images[key] = copy.deepcopy(cropped_rgb)  # Store the full resolution cropped image
        except queue.Empty:
            input(
                f"{key} camera frozen. Check connect, then press enter to relaunch..."
            )
            cap.close()
            init_cameras(caps, config["REALSENSE_CAMERAS"])
            return get_im(caps, config, observation_space, save_video, display_image, img_queue, recording_frames)

    # Store full resolution cropped images separately
    if save_video:
        recording_frames.append(full_res_images)

    if display_image:
        img_queue.put(display_images)
    return images

def init_cameras(name_serial_dict):
    """Init both wrist cameras."""
    caps = {}
    for cam_name, serial_number in name_serial_dict.items():
        caps[cam_name] = VideoCapture(RSCapture(name=cam_name, serial_number=serial_number))
    return caps

def main():
    # Configurazione delle telecamere
    config = {
        "REALSENSE_CAMERAS": {
            "wrist_1": "130322273284"  # Sostituisci con il numero di serie della tua telecamera
        },
        "IMAGE_CROP": {
            "wrist_1": lambda img: crop_image(img, crop_size=(100, 100)) #edit per croppare
        }
    }
    observation_space = {
        "images": {
            "wrist_1": np.zeros((300, 300, 3), dtype=np.uint8) 
        }
    }

    # Inizializza le telecamere
    caps = init_cameras(config["REALSENSE_CAMERAS"])

    try:
        while True:
            # Acquisisci le immagini dalle telecamere
            images = get_im(caps, config, observation_space)

            # Visualizza le immagini utilizzando OpenCV
            for key, img in images.items():
                cv2.imshow(f"{key} Camera", img)

            # Esci dal loop se viene premuto il tasto 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Aggiungi un ritardo per limitare la frequenza di acquisizione delle immagini
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Interruzione da tastiera ricevuta. Uscita...")

    finally:
        # Chiudi le telecamere e le finestre di visualizzazione
        for cap in caps.values():
            cap.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()