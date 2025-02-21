export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd_sim.py "$@" \
    --exp_name=pick_cube_sim \
    --checkpoint_path=first_run \
    --demo_path=../../demo_data/pick_cube_sim_3_demos_2024-12-10_15-07-37.pkl\
    --learner \
    
    
    
