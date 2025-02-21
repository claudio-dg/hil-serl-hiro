from pathlib import Path
import mujoco
import numpy as np
from mujoco.glfw import glfw

# Carica il modello MuJoCo
model_path = Path("/home/claudiodelgaizo/ros/catkin_ws/src/hil-serl/franka_sim/franka_sim/envs/xmls/scene.xml")
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Inizializza il contesto di rendering
glfw.init()
window = glfw.create_window(1200, 900, "MuJoCo Simulation", None, None)
glfw.make_context_current(window)

# Inizializza il contesto di visualizzazione
scene = mujoco.MjvScene(model, maxgeom=10000)
camera = mujoco.MjvCamera()
option = mujoco.MjvOption()
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Imposta la telecamera
camera.azimuth = 90
camera.elevation = -20
camera.distance = 2.0
camera.lookat = np.array([0.0, 0.0, 0.0])

# Funzione di rendering
def render():
    mujoco.mj_step(model, data)
    viewport = mujoco.MjrRect(0, 0, glfw.get_framebuffer_size(window)[0], glfw.get_framebuffer_size(window)[1])
    mujoco.mjv_updateScene(model, data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    glfw.swap_buffers(window)

# Loop principale
while not glfw.window_should_close(window):
    glfw.poll_events()
    render()

# Cleanup
glfw.terminate()