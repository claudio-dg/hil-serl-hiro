import sys
import mujoco as mj
import glfw
import mujoco.viewer

# Variabili globali per MuJoCo
m = None  # Modello
d = None  # Dati della simulazione
cam = mj.MjvCamera()  # Camera
opt = mj.MjvOption()  # Opzioni di visualizzazione

# Variabili per interazione con il mouse
# button_left = False
# button_middle = False
# button_right = False
# lastx, lasty = 0, 0


# def keyboard(window, key, scancode, act, mods):
#     """Callback per la tastiera: resetta la simulazione con BACKSPACE."""
#     if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
#         mj.mj_resetData(m, d)
#         mj.mj_forward(m, d)
#         # print("\n AAAAAAAAAAAAAAAAAAAAAA \n")


# def mouse_button(window, button, act, mods):
#     """Callback per il mouse: aggiorna lo stato dei tasti e la posizione del cursore."""
#     global button_left, button_middle, button_right, lastx, lasty
#     button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
#     button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
#     button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
#     lastx, lasty = glfw.get_cursor_pos(window)
#     # print("\n AAAAAAAAAAAAAAAAAAAAAA \n")


# def mouse_move(window, xpos, ypos):
#     """Callback per il movimento del mouse: ruota o sposta la camera."""
#     global lastx, lasty
#     # print("\n AAAAAAAAAAAAAAAAAAAAAA \n")
#     if not (button_left or button_middle or button_right):
#         return

#     dx = xpos - lastx
#     dy = ypos - lasty
#     lastx, lasty = xpos, ypos

#     width, height = glfw.get_window_size(window)
#     mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or \
#                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

#     if button_right:
#         action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
#     elif button_left:
#         action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
#     else:
#         action = mj.mjtMouse.mjMOUSE_ZOOM

#     mj.mjv_moveCamera(m, action, dx / height, dy / height, scn, cam)


# def scroll(window, xoffset, yoffset):
#     """Callback per lo scroll: zoom della camera."""
#     mj.mjv_moveCamera(m, mj.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoffset, scn, cam)


def main():
    global m, d

    # Controllo degli argomenti
    if len(sys.argv) < 2:
        print("missing argument: path to the xml_file : USAGE: python script.py modelfile")
        return

    model_file = sys.argv[1]

    # Caricamento del modello
    try:
        m = mj.MjModel.from_xml_path(model_file)
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        return

    # Creazione della struttura dati
    d = mj.MjData(m)

     # Stampiamo il numero di DOF attesi
    print(f"Dimensione attesa di qpos: {m.nq}")

    # # Se il cubo ha un freejoint, dobbiamo inizializzarlo correttamente
    # if m.nq > 13:
    #     d.qpos[13:] = [0.5, 0, 0.05, 1, 0, 0, 0]  # Posizione e quaternione del cubo



    # Avvia la finestra MuJoCo Viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mj.mj_step(m, d)  # Avanza la simulazione
            viewer.sync()  # Aggiorna la finestra

    print("Simulazione terminata.")


if __name__ == "__main__":
    main()
