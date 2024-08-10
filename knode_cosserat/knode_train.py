import torch
import torch.nn as nn
import numpy as np
import tqdm
from itertools import chain
import sys
from scipy.optimize import fsolve, minimize
import matplotlib.pyplot as plt
import argparse
from knode import setup_robot, simulate
from fastdtw import fastdtw

from cosserat_ode_torch import CosseratRodTorch
from cosserat_ode import CosseratRod

if torch.cuda.is_available():
    device = "cuda" # Use CUDA if possible
else:
    device = "cpu" # For Mac GPU
# MODEL_SAVE_PATH = "saved_models/one_layer_512_tanh_sine_10_30_45_trainlen_25_3000_epoch.pth"
TRAIN = True  # whether to train the model. If False, only compute the loss
CLAMP_WEIGHT = True
RESUME_TRAINING = False
EPOCHS = 4000

#####################################################
################       Data       ###################
#####################################################
# appending multiple trajectories into one list as training data
train_len = 150  # length of training data
batch_len = 150

parser = argparse.ArgumentParser(description='Train KNODE.')
parser.add_argument('--air', action=argparse.BooleanOptionalAction, default=True, help='model air resistance')
parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('data_path', type=str, help='data path')
args = parser.parse_args()

data_path_ls = [args.data_path]
data_short = '-'.join(s.split('/')[-1].replace('.bag.npy', '').replace('_1000_steps_10_grid', '') for s in data_path_ls)
MODEL_SAVE_PATH = f'saved_models/{data_short}+{args.air}_trainlen_{train_len}_{EPOCHS}_epoch.pth'
print(data_path_ls, MODEL_SAVE_PATH)

# extracting training data
np_datas = [np.load(data_path, allow_pickle=True).item() for data_path in data_path_ls]

robot_nonn = CosseratRod(use_fsolve=True)
setup_robot(robot_nonn, args.air)

np_traj_ls = None
def forward_datas(torch_robot=None, mix=0):
    global torch_traj_ls, np_traj_ls, torch_controls_ls
    torch_traj_ls = []
    torch_controls_ls = []
    new_trajs = []
    if torch_robot is not None:
        nn_model = torch_robot.nn_models
        param_ls = []
        for _, layer in nn_model.state_dict().items():
            param_ls.append(layer.detach().cpu().numpy())
        robot_nonn.nn_model = nn_model
        robot_nonn.param_ls = param_ls
        robot_nonn.nn_path = 'whatever' # Force the robot to use nn
    for i, np_data in enumerate(np_datas):
        controls_np = np_data['controls'][:train_len]

        # only take the first 25 elements. The rest are history terms
        traj_np = simulate(robot_nonn, controls_np[:train_len])[:train_len, :25]
        # print('MSE', np.mean((traj_np[:,:3,[3,5,7,9]] - np_data['interpolated'][:train_len,:3,[3,5,7,9]])**2))
        # for j in range(10):
            # print('DTW Distance XYZ', fastdtw(traj_np[:,:3,j], np_data['interpolated'][:train_len,:3,j])[0])
            # print('Error', np.mean((traj_np[:,:3,j] - np_data['interpolated'][:train_len,:3,j])**2))
        dtw_metric = fastdtw(traj_np[:,:3,9], np_data['interpolated'][:train_len,:3,9])[0]
        dtw_arr.append(dtw_metric)
        print('DTW Distance XYZ', dtw_metric)

        # METHOD 1
        if np_traj_ls is not None:
            traj_np = 0.1 * traj_np + 0.9 * np_traj_ls[i]

        traj_np[:train_len, :3, 3] = np_data['interpolated'][:train_len, :3, 3]
        traj_np[:train_len, :3, 5] = np_data['interpolated'][:train_len, :3, 5]
        traj_np[:train_len, :3, 7] = np_data['interpolated'][:train_len, :3, 7]
        traj_np[:train_len, :3, 9] = np_data['interpolated'][:train_len, :3, 9]

        # OTHER METHODS
        # traj_np[:train_len, :10] = np_data['estimated'][:train_len, :10]
        # traj_np[:train_len, 19:] = np_data['estimated'][:train_len, 19:]
        # traj_np[:train_len] = np_data['estimated'][:train_len]

        # METHOD 2
        # traj_np = mix * traj_np + (1 - mix) * np_data['estimated'][:train_len]

        controls = torch.tensor(controls_np).float().to(device)
        traj = torch.tensor(traj_np, requires_grad=True).float().to(device)

        torch_traj_ls.append(traj)
        new_trajs.append(traj_np)
        torch_controls_ls.append(controls)
    np_traj_ls = new_trajs

forward_datas()
print("Total number of trajectories: ", len(torch_traj_ls))
print("training trajectory has shape: ", np_traj_ls[0].shape)
print("training control has shape: ", torch_controls_ls[0].shape)


#####################################################
################       Robot       ##################
#####################################################
# fix all random seeds for pytorch
torch.manual_seed(0)

robot = CosseratRodTorch(device)
setup_robot(robot)

# load the model from a checkpoint
if RESUME_TRAINING:
    checkpoint = torch.load(MODEL_SAVE_PATH)
    robot = checkpoint['robot']

robot.use_nn = True

#####################################################
###############       Training       ################
#####################################################
# Training parameters
loss_arr = []
dtw_arr = []
param = robot.nn_models.parameters()
optimizer = torch.optim.Adam(param, lr=0.01, weight_decay=1e-3)

# load optimizer and loss array if training from a checkpoint
if RESUME_TRAINING:
    # optimizer.load_state_dict(checkpoint['optim'])
    loss_arr = checkpoint['loss']

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=90, factor=0.8)
loss_func = nn.MSELoss()

progress = tqdm.tqdm(range(EPOCHS))
for epoch in progress:
    # reset the losses
    grow_loss = 0
    total_loss = 0

    for traj_idx in range(len(torch_traj_ls)):

        traj = torch_traj_ls[traj_idx]
        controls = torch_controls_ls[traj_idx]

        ys = traj[:batch_len-1, 0:19, :]
        zs = traj[:batch_len-1, 19:, :]

        y_prevs = torch.cat((ys[:1], ys[:-1]))
        z_prevs = torch.cat((zs[:1], zs[:-1]))

        # Set the guess to the next state, which should give 0 residual
        Gs = traj[1:batch_len]

        # Compute the guess for the next segment
        key_pt_idx = np.array([3, 5, 7, 9]) # 9 being the tip of the rod
        grow_trajs = robot.parallelGetNextSegmentEuler(Gs, key_pt_idx, {
            "yh": robot.c1 * ys + robot.c2 * y_prevs,
            "zh": robot.c1 * zs + robot.c2 * z_prevs,
            "tendon_tensions": controls[:batch_len - 1],
        })
        for batch_idx in range(batch_len-1):
            grow_traj = grow_trajs[batch_idx]
            grow_loss_deltas = [
                loss_func(grow_traj[:3], traj[batch_idx+1][:3, key_pt_idx]),
                # loss_func(grow_traj[:25], traj[batch_idx+1][:25, key_pt_idx])
                # loss_func(grow_traj[3:7], traj[batch_idx+1][3:7, key_pt_idx]),
                # 1e-6 * loss_func(grow_traj[7:10], traj[batch_idx+1][7:10, key_pt_idx]),
                # 1e-4 * loss_func(grow_traj[10:13], traj[batch_idx+1][10:13, key_pt_idx]),
                # loss_func(grow_traj[:7], traj[batch_idx+1][:7, key_pt_idx]),
                # loss_func(grow_traj[13:19], traj[batch_idx+1][13:19, key_pt_idx]),
                # loss_func(grow_traj[19:25], traj[batch_idx+1][19:25, key_pt_idx-1])
            ]
            # if epoch % 10 == 0:
                # print([x.item() for x in grow_loss_deltas])

            grow_loss += sum(grow_loss_deltas)

    total_loss = grow_loss
    total_loss /= (batch_len-1)  # average loss over all time steps
    loss_arr.append(total_loss.item())

    if epoch % 10 == 0 and args.verbose:
        print(f"Epoch {epoch} of {EPOCHS}")
        print(f"Total loss: {total_loss}, lr {scheduler.get_last_lr()}")
    progress.set_description(f"loss: {total_loss:.2E}, lr {scheduler.get_last_lr()}")


    if epoch % 200 == 0 and epoch != 0:
        # try:
            # forward_datas(robot)
        # except:
            # print('oops')
        mix=min(epoch/500, 1)
        # print('Re-simulating', mix)
        forward_datas(robot, mix=mix)
        for g in optimizer.param_groups:
            g['lr'] *= 1.1
    #     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.8)

    if not TRAIN:
        break
    else:
        # save the model periodically
        if epoch % 500 == 0 and epoch != 0:
            print("saving model")
            torch.save({'robot': robot,
                        'loss': loss_arr,
                        'dtw': dtw_arr,
                        'optim': optimizer.state_dict()},
                        MODEL_SAVE_PATH)
    optimizer.zero_grad()
    total_loss.backward()

        # Check gradients
        # for name, param in robot.nn_models.named_parameters():
        #         print(f"Parameter: {name}, Gradient: {param.grad}")

    optimizer.step()
    scheduler.step(total_loss)

    if CLAMP_WEIGHT:
        # clamp the weights to be non-negative
        for name, param in robot.nn_models.named_parameters():
            if 'weight' in name and "layer1" not in name:
                with torch.no_grad():
                    param.clamp_(min=0)

# save the model
if MODEL_SAVE_PATH is not None and TRAIN:
    torch.save({'robot': robot,
                'loss': loss_arr,
                'optim': optimizer.state_dict()},
                MODEL_SAVE_PATH)
