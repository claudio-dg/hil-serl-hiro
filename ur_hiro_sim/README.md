
# Installation:
- From `hil-serl-hiro` folder, cd into `ur_hiro_sim`.
- In your `hil-serl` conda environment, run `pip install -e .` to install this package.
- run `pip install -r requirements.txt` to install sim dependencies.


# Notes:
- Error due to `egl` when running on a CPU machine:
```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```
