# Continuum Robot

  - `physics_train.py`: Training
  - `physics_multitrain.py`: Mutli-train and multi-evaluation
  - `physics_train.py`: Experiments on training on simulated trajectories instead of real-world data

The `*.bag` files from physical experiments should be placed at a folder called `physical_experiment_data` next to the python files.

## Training Flow

1. Run `knode.py physical_experiment_data/....bag` to generate a data file for the trajectory in `datas/....npy`
2. Run `knode_train.py datas/....npy`. It prints out the location of the saved model on the first line.
3. Run `knode.py physical_experiment_data/....bag --model saved_models/....pth` to evaluate the saved model and print the DTW distance
