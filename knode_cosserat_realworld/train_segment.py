import torch
import torch.nn as nn
import numpy as np
from cosserat_ode_torch import CosseratRodTorch
import tqdm
from Utils.data_processing import normalize_data
from itertools import chain
from Utils.transformations import quaternion_to_euler
import argparse

parser = argparse.ArgumentParser(description='Train KNODE.')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--layers', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=1e-1)
parser.add_argument('--train_len', type=int, default=120)
parser.add_argument('--save_path', type=str, default="saved_models/quick_test.pth")
parser.add_argument('--noise_traj', type=float, default=0.01)
parser.add_argument('--noise_controls', type=float, default=0)
parser.add_argument('--data', type=str, default='sinesine')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

device = "cpu"
MODEL_SAVE_PATH = args.save_path
TRAIN = True  # whether to train the model. If False, only compute the loss
CLAMP_WEIGHT = True
RESUME_TRAINING = False
EPOCHS = args.epochs

#####################################################
################       Data       ###################
#####################################################
# appending multiple trajectories into one list as training data
train_len = args.train_len  # length of training data
trim_len = 100 # trim the first few steps to avoid steps without any movements
batch_len = args.train_len

# data_path_ls = ['data/knode/sine10_traj_1000_steps_10_grid_E209.npy',
#                 'data/knode/sine30_traj_1000_steps_10_grid_E209.npy',
#                 'data/knode/sine45_traj_1000_steps_10_grid_E209.npy']

# data_path_ls = ['data/real_param_sine10_traj_100_steps_10_grid_damp_3e-2.npy',
#                 'data/real_param_sine35_traj_100_steps_10_grid_damp_3e-2.npy',
#                 'data/real_param_sine40_traj_100_steps_10_grid_damp_3e-2.npy']

data_path_ls = ['data/real_physical/sin_1_0_amp_300_estimated.npy'] # first 50 steps are already trimmed
if args.data == 'sinesine':
    data_path_ls = [
        'data/real_physical/sin_1_0_amp_300_estimated.npy',
        'data/real_physical/sin_3_0_amp_300_estimated.npy',
    ]
if args.data == 'sinesinerand':
    data_path_ls = [
        'data/real_physical/sin_1_0_amp_300_estimated.npy',
        'data/real_physical/sin_3_0_amp_300_estimated.npy',
        'data/real_physical/rand_0_60s_estimated.npy',
    ]
if args.data == 'sinesinestep':
    data_path_ls = [
        'data/real_physical/sin_1_0_amp_300_estimated.npy',
        'data/real_physical/sin_3_0_amp_300_estimated.npy',
        'data/real_physical/dir_a_tension_950_estimated.npy',
    ]
if args.data == 'sinesinestepstep':
    data_path_ls = [
        'data/real_physical/sin_1_0_amp_300_estimated.npy',
        'data/real_physical/sin_3_0_amp_300_estimated.npy',
        'data/real_physical/dir_a_tension_950_estimated.npy',
        'data/real_physical/dir_a_tension_1250_estimated.npy',
    ]

# extracting training data
torch_traj_ls = []
torch_controls_ls = []

# fix all random seeds for pytorch
torch.set_num_threads(1)
torch.manual_seed(args.seed)

for data_path in data_path_ls:
    np_data = np.load(data_path, allow_pickle=True).item()
    traj_np = np_data['traj'][trim_len:train_len+trim_len, :25] # only take the first 25 elements. The rest are history terms
    controls_np = np_data['controls'][trim_len:train_len+trim_len]

    # add noise to the data
    traj = torch.tensor(traj_np, requires_grad=True).float().to(device) + torch.randn(traj_np.shape).float().to(device) * args.noise_traj
    controls = torch.tensor(controls_np).float().to(device) + torch.randn(controls_np.shape).float().to(device) * args.noise_controls

    torch_traj_ls.append(traj)
    torch_controls_ls.append(controls)

print("Total number of trajectories: ", len(torch_traj_ls))
print("training trajectory has shape: ", traj_np.shape)
print("training control has shape: ", controls_np.shape)

#####################################################
################       Robot       ##################
#####################################################

robot = CosseratRodTorch(args.layers)

# load the model from a checkpoint
if RESUME_TRAINING:
    checkpoint = torch.load("saved_models/one_layer_1024_tanh_random_20_trainlen_2000_epoch.pth")
    robot = checkpoint['robot']

robot.use_nn = True

#####################################################
###############       Training       ################
#####################################################
# Training parameters
loss_arr = []
param = robot.nn_models.parameters()
optimizer = torch.optim.Adam(param,
                             lr=1e-2,
                             weight_decay=args.weight_decay)

# load optimizer and loss array if training from a checkpoint
if RESUME_TRAINING:
    optimizer.load_state_dict(checkpoint['optim'])
    loss_arr = checkpoint['loss']

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 'min',
                                                 patience=80,
                                                 factor=0.5,
                                                 verbose=True)
loss_func = nn.MSELoss()

for epoch in tqdm.tqdm(range(EPOCHS)):
    # reset the losses
    grow_loss = 0
    total_loss = 0

    for traj_idx in range(len(torch_traj_ls)):
        traj = torch_traj_ls[traj_idx]
        controls = torch_controls_ls[traj_idx]

        for stp_idx in range(batch_len-1):
            batch_idx = ((epoch % batch_len - 1) * batch_len + stp_idx + batch_len) % train_len
            if batch_idx >= train_len-1:
                break
            # print("current index: ", batch_idx)
            y = traj[batch_idx, 0:19, :]
            z = traj[batch_idx, 19:, :]
            if stp_idx == 0:  # should not have i==0, otherwise loss for i!= 0 is not zero
                # for i=0, j=0, set the prev state to be the current state
                y_prev = y.clone().requires_grad_(True)
                z_prev = z.clone().requires_grad_(True)
            else:
                y_prev = traj[batch_idx-1, 0:19, :]
                z_prev = traj[batch_idx-1, 19:, :]

            robot.y = y  # Updating the robot state to y
            robot.z = z

            # Set the guess to the next state, which should give 0 residual
            G = torch.cat((traj[batch_idx+1, :19, :], traj[batch_idx+1, 19:, :])).clone().requires_grad_(True)
            #print("train tendon_tensions: ", controls[batch_idx])

            robot.tendon_tensions = controls[batch_idx]  # Control inputs

            # Set history terms - Eq(5)
            yh = robot.c1 * robot.y + robot.c2 * y_prev
            zh = robot.c1 * robot.z + robot.c2 * z_prev

            robot.residualArgs["yh"] = yh
            robot.residualArgs["zh"] = zh
            grow_traj = robot.getNextSegmentEuler(G)
            key_pt_idx = torch.tensor([1, 3, 6, 9]).to(device) # 9 being the tip of the rod
            grow_loss_delta = loss_func(grow_traj[:3, key_pt_idx],
                                    traj[batch_idx+1][:3, key_pt_idx]) + \
                            loss_func(grow_traj[7:19, key_pt_idx],
                                    traj[batch_idx+1][7:19, key_pt_idx]) + \
                            loss_func(quaternion_to_euler(grow_traj[3:7, key_pt_idx]),
                                    quaternion_to_euler(traj[batch_idx+1][3:7, key_pt_idx])) +\
                            loss_func(grow_traj[19:, key_pt_idx],
                                    traj[batch_idx+1][19:, key_pt_idx-1])

            grow_loss += grow_loss_delta

    total_loss = grow_loss
    total_loss /= (batch_len-1)  # average loss over all time steps
    loss_arr.append(total_loss.item())

    if epoch % 10 == 0:
        print(f"\nEpoch {epoch} of {EPOCHS}")
        print(f"\nTotal loss: {total_loss}")

    if not TRAIN:
        break
    else:
        # save the model periodically
        if epoch % 50 == 0 and epoch != 0:
            print("\nsaving model")
            torch.save({'robot': robot,
                        'loss': loss_arr,
                        'optim': optimizer.state_dict()},
                        MODEL_SAVE_PATH)
    optimizer.zero_grad()
    total_loss.backward()


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
