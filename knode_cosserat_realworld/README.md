# Continuum Robot, Real worl data

 - `prepare.py`: Generates data files for training
 - `estamate_state.py`: Generates more data files for training

    You need to run both `prepare.py` and `estimated_state.py` to produce all the data files necessary. For example:

    ```bash
    python prepare.py physical_experiment_data/dir_a_tension_1100.bag
    python estimate_state.py dir_a_tension_1100
    ```

    You will need to run this for each bag file in the `physical_experiment_data` directory.

 - `simulate.py`: Evaluation
 - `train_segment.py`: Training
