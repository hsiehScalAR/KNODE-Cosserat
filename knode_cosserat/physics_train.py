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
from physics_controls import calc_controls
from scipy.spatial.transform import Rotation
import io

from cosserat_ode_torch import CosseratRodTorch
from cosserat_ode import CosseratRod
from Utils.transformations import quaternion_to_euler

if torch.cuda.is_available():
    device = "cuda" # Use CUDA if possible
else:
    device = "cpu"
# MODEL_SAVE_PATH = "saved_models/one_layer_512_tanh_sine_10_30_45_trainlen_25_3000_epoch.pth"
TRAIN = True  # whether to train the model. If False, only compute the loss
CLAMP_WEIGHT = True
RESUME_TRAINING = False

#####################################################
################       Data       ###################
#####################################################
# appending multiple trajectories into one list as training data
train_len = 30  # length of training data
batch_len = 30
eval_len = 100

parser = argparse.ArgumentParser(description='Train KNODE.')
parser.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--original', action=argparse.BooleanOptionalAction, default=False, help="use original parameters")
parser.add_argument('--mod', type=str, default=None)
parser.add_argument('control_type_arg', nargs='+', type=str, help='Trajectories to train on. For example "sine 2"')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--noise_traj', type=float, default=0)
parser.add_argument('--noise_controls', type=float, default=0)
parser.add_argument('--layers', type=float, default=512)
parser.add_argument('--validation', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--fast', action=argparse.BooleanOptionalAction, default=False, help="use fast but inaccurate training")

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

args = parser.parse_args()
control_type, control_arg = split_list(args.control_type_arg)
control_arg = [float(i) for i in control_arg]

if args.validation is None:
    args.validation = 'sine 0.1' if args.original else 'sine 1.25'

validation_type, validation_arg = args.validation.split(' ')
validation_arg = float(validation_arg)

saves = {}

prefix = "physics_original" if args.original else "physics"
data_short = f'{prefix}_{"-".join(control_type)}_{"-".join(map(str, control_arg))}'.replace('.', '_')
MODEL_SAVE_PATH = f'saved_models/{data_short}_{args.mod}_trainlen_{train_len}_{args.epochs}_epoch_{args.seed}.pth'
print(MODEL_SAVE_PATH)

# For creating the initial trajectory to train on
robot_reference = CosseratRod(use_fsolve=True)
setup_robot(robot_reference, original=args.original)

# For evaluating the model
robot_eval = CosseratRod(use_fsolve=True)
setup_robot(robot_eval, args.mod, args.original)

def compute_references():
    if len(control_type) != len(control_arg):
        raise Exception('Different number of control_type and control_arg')
    for i in range(len(control_type)):
        controls = np.array(calc_controls(control_type[i], control_arg[i], robot_reference.del_t, eval_len))
        reference_traj = simulate(robot_reference, controls)[:, :25]
        yield reference_traj

def compute_validation_reference():
    controls = np.array(calc_controls(validation_type, validation_arg, robot_reference.del_t, eval_len))
    return simulate(robot_reference, controls)[:, :25]

# references = list(compute_references())
validation_reference = compute_validation_reference()

np_traj_ls = None
reference_traj = None
def forward_datas(torch_robot=None):
    global torch_traj_ls, np_traj_ls, torch_controls_ls, reference_traj
    torch_traj_ls = []
    torch_controls_ls = []
    new_trajs = []
    if torch_robot is not None:
        nn_model = torch_robot.nn_models
        param_ls = []
        for _, layer in nn_model.state_dict().items():
            param_ls.append(layer.detach().cpu().numpy())
        robot_reference.nn_model = nn_model
        robot_reference.param_ls = param_ls
        robot_reference.nn_path = 'whatever' # Force the robot to use nn
    if len(control_type) != len(control_arg):
        raise Exception('Different number of control_type and control_arg')
    for i in range(len(control_type)):
        controls = calc_controls(control_type[i], control_arg[i], robot_reference.del_t, train_len)
        controls_np = np.array(controls)
        reference_traj = simulate(robot_reference, controls_np)[:, :25]
        traj_np = reference_traj

        # Do the interpolation
        # idx = [0, 3, 5, 7, 9]
        # traj_np = np.array([interpolate_posquat(t[:3,idx].T, [Rotation.from_quat(q, scalar_first=True) for q in t[3:7,idx].T], 10) for t in traj_np])
        # traj_np = estimate_state(traj_np[:,:7, :], robot_reference)

        # controls = torch.tensor(controls_np).float().to(device)
        # traj = torch.tensor(traj_np, requires_grad=True).float().to(device)
        traj = torch.tensor(traj_np, requires_grad=True).float().to(device) + torch.randn(traj_np.shape).float().to(device) * args.noise_traj
        controls = torch.tensor(controls_np).float().to(device) + torch.randn(controls_np.shape).float().to(device) * args.noise_controls
        # plt.plot(traj_np[:,:3,9])
        # plt.show()

        torch_traj_ls.append(traj)
        new_trajs.append(traj_np)
        torch_controls_ls.append(controls)
    np_traj_ls = new_trajs

def evaluate(torch_robot=None):
    if torch_robot is not None:
        nn_model = torch_robot.nn_models
        param_ls = []
        for _, layer in nn_model.state_dict().items():
            param_ls.append(layer.detach().cpu().numpy())
        robot_eval.nn_model = nn_model
        robot_eval.param_ls = param_ls
        robot_eval.nn_path = 'whatever' # Force the robot to use nn
    if len(control_type) != len(control_arg):
        raise Exception('Different number of control_type and control_arg')
    dtw_metrics = []
    # for i in range(len(control_type)):
    #     controls = calc_controls(control_type[i], control_arg[i], robot_reference.del_t, eval_len)
    #     controls_np = np.array(controls)
    #     traj_np = simulate(robot_eval, controls_np[:eval_len])[:eval_len, :25]
    #     dtw_metric = fastdtw(traj_np[:,:3,9], references[i][:,:3,9])[0]
    #     dtw_metrics.append(dtw_metric)
    #     print('DTW Distance XYZ', dtw_metric)

    controls = calc_controls(validation_type, validation_arg, robot_reference.del_t, eval_len)
    controls_np = np.array(controls)
    traj_np = simulate(robot_eval, controls_np[:eval_len])[:eval_len, :25]
    dtw_metric = fastdtw(traj_np[:,:3,9], validation_reference[:,:3,9])[0]
    dtw_metrics.append(dtw_metric)
    print('Validation DTW Distance XYZ', dtw_metric)

    dtw_arr.append(dtw_metrics)
    buff = io.BytesIO()
    torch.save({ 'robot': robot }, buff)
    buff.seek(0)
    saves[dtw_metric] = buff

forward_datas()
print("Total number of trajectories: ", len(torch_traj_ls))
print("training trajectory has shape: ", np_traj_ls[0].shape)
print("training control has shape: ", torch_controls_ls[0].shape)


#####################################################
################       Robot       ##################
#####################################################
# fix all random seeds for pytorch
torch.set_num_threads(1)
torch.manual_seed(args.seed)

robot = CosseratRodTorch(device, args.layers)
setup_robot(robot, args.mod, args.original)

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
optimizer = torch.optim.Adam(param, lr=1e-2, weight_decay=args.weight_decay)

# load optimizer and loss array if training from a checkpoint
if RESUME_TRAINING:
    # optimizer.load_state_dict(checkpoint['optim'])
    loss_arr = checkpoint['loss']

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=80, factor=0.5)
loss_func = nn.MSELoss()

if not args.fast:
    for epoch in tqdm.tqdm(range(args.epochs + 1)):
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
                key_pt_idx = torch.tensor([2, 6, 9]).to(device) # 9 being the tip of the rod

                grow_loss_delta = loss_func(grow_traj[:3, key_pt_idx],
                                        traj[batch_idx+1][:3, key_pt_idx]) + \
                                loss_func(grow_traj[7:19, key_pt_idx],
                                        traj[batch_idx+1][7:19, key_pt_idx]) + \
                                loss_func(quaternion_to_euler(grow_traj[3:7, key_pt_idx]),
                                        quaternion_to_euler(traj[batch_idx+1][3:7, key_pt_idx])) +\
                                loss_func(grow_traj[19:, key_pt_idx],
                                        traj[batch_idx+1][19:, key_pt_idx-1])

                # grow_loss_delta = loss_func(grow_traj[:3, key_pt_idx], traj[batch_idx+1][:3, key_pt_idx])


                grow_loss += grow_loss_delta

        total_loss = grow_loss
        total_loss /= (batch_len-1)  # average loss over all time steps
        loss_arr.append(total_loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch} of {args.epochs}")
            print(f"Total loss: {total_loss}, lr {scheduler.get_last_lr()}")

        if epoch % 50 == 0 and args.eval:
            evaluate(robot if epoch != 0 else None)


        if not TRAIN:
            break
        else:
            # save the model periodically
            if epoch % 50 == 0 and epoch != 0:
                print("saving model")
                torch.save({'robot': robot,
                            'dtw': dtw_arr,
                            'loss': loss_arr,
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

else:
    progress = tqdm.tqdm(range(args.epochs + 1))
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
            # grow_trajs = [robot.getNextSegmentEuler(Gs[i], {
            #     "yh": robot.c1 * ys[i] + robot.c2 * y_prevs[i],
            #     "zh": robot.c1 * zs[i] + robot.c2 * z_prevs[i],
            #     "tendon_tensions": controls[i]
            # })[:, key_pt_idx] for i in range(batch_len-1)]

            for batch_idx in range(batch_len-1):
                grow_traj = grow_trajs[batch_idx]
                # print(grow_traj - traj[batch_idx+1][:, key_pt_idx])


                grow_loss_delta = loss_func(grow_traj[:3],
                                        traj[batch_idx+1][:3, key_pt_idx]) + \
                                loss_func(grow_traj[7:19],
                                        traj[batch_idx+1][7:19, key_pt_idx]) + \
                                loss_func(quaternion_to_euler(grow_traj[3:7]),
                                        quaternion_to_euler(traj[batch_idx+1][3:7, key_pt_idx])) +\
                                loss_func(grow_traj[19:],
                                        traj[batch_idx+1][19:, key_pt_idx-1])


                # grow_loss_deltas = [
                #     loss_func(grow_traj[:3], traj[batch_idx+1][:3, key_pt_idx]),
                #     loss_func(grow_traj[3:7], traj[batch_idx+1][3:7, key_pt_idx]),
                #     # loss_func(grow_traj[:7], traj[batch_idx+1][:7, key_pt_idx]),
                #     # loss_func(grow_traj[13:19], traj[batch_idx+1][13:19, key_pt_idx]),
                #     loss_func(grow_traj[19:], traj[batch_idx+1][19:, key_pt_idx-1])
                # ]
                # if epoch % 10 == 0:
                    # print([x.item() for x in grow_loss_deltas])

                grow_loss += grow_loss_delta

        total_loss = grow_loss
        total_loss /= (batch_len-1)  # average loss over all time steps
        loss_arr.append(total_loss.item())

        if epoch % 10 == 0 and args.verbose:
            print(f"Epoch {epoch} of {args.epochs}")
            # print(f"Total loss: {total_loss}, lr {scheduler.get_last_lr()}")
            print(f"Total loss: {total_loss}")
        #progress.set_description(f"loss: {total_loss:.2E}, lr {scheduler.get_last_lr()}")
        progress.set_description(f"loss: {total_loss:.2E}")


        if epoch % 200 == 0 and args.eval:
            evaluate(robot if epoch != 0 else None)

        if not TRAIN:
            break
        else:
            # save the model periodically
            if epoch % 500 == 0 and epoch != 0:
                print("saving model")
                torch.save({'robot': robot,
                            'dtw': dtw_arr,
                            'loss': loss_arr,
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

    if args.eval:
        best_dtw = min(saves.keys())
        print('Saving model with dtw', best_dtw)
        torch.save({ **torch.load(saves[best_dtw]),
                    'dtw': dtw_arr,
                    'loss': loss_arr,
                    'optim': optimizer.state_dict()},
                   MODEL_SAVE_PATH)
