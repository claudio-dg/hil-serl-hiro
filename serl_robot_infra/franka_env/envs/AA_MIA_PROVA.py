import sys
import os

# Aggiungi il percorso del modulo franka_env al PYTHONPATH
sys.path.append("/home/claudiodelgaizo/ros/catkin_ws/src/hil-serl/serl_robot_infra/franka_env")

from camera.video_capture import VideoCapture
from camera.rs_capture import RSCapture

# from franka_env.camera.video_capture import VideoCapture
# from franka_env.camera.rs_capture import RSCapture

import cv2
import time

def main():
    # Inizializza la telecamera RealSense
    camera_name = "wrist_1"
    serial_number = "130322273284"  # Sostituisci con il numero di serie della tua telecamera
    cap = VideoCapture(RSCapture(name=camera_name, serial_number=serial_number))

    try:
        while True:
            # Acquisisci un'immagine dalla telecamera
            rgb = cap.read()

            # Visualizza l'immagine utilizzando OpenCV
            cv2.imshow("RealSense Camera", rgb)

            # Esci dal loop se viene premuto il tasto 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Aggiungi un ritardo per limitare la frequenza di acquisizione delle immagini
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Interruzione da tastiera ricevuta. Uscita...")

    finally:
        # Chiudi la telecamera e la finestra di visualizzazione
        cap.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()