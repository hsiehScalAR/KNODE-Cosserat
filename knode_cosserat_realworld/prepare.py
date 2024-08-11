import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.animation as animation
from scipy.signal import savgol_filter
import os.path
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation
import torch
import argparse

from preprocess import preprocessed

# Number of seconds to trip from each bag file
to_trim = {
    'physical_experiment_data/dir_a_tension_800.bag': 2,
    'physical_experiment_data/dir_a_tension_950.bag': 0,
    'physical_experiment_data/dir_a_tension_1100.bag': 9,
    'physical_experiment_data/dir_a_tension_1250.bag': 8,
    'physical_experiment_data/dir_a_tension_1400.bag': 6,
}

plt.rcParams['svg.fonttype'] = 'none'

import sys
sys.path.append('../Soft-KNODE')
from scipy.optimize import fsolve, minimize
from cosserat_ode import CosseratRod

def setup_robot_original(robot, mod=None):
    """Set up robot based on original parametrers"""
    robot.del_t = 0.005
    robot.L = 0.4 # length
    robot.E = 209e9 # Young's modulus
    robot.r = 0.0012 # rod radius
    robot.rho = 8000. # rod density

    Bbt = 5e-4

    if mod == None:
        pass
    elif mod == 'nsw':
        if isinstance(robot, CosseratRod):
            robot.g = np.array([0, 0, 0])
        else:
            robot.g = torch.tensor([0, 0, 0], device=robot.device)
    elif mod == 'short':
        robot.L = 0.3
    elif mod == 'damping':
        Bbt = 9e-4
    elif mod == 'diameter':
        robot.r = 0.002
    elif mod == 'youngs':
        robot.E = 109e9
    elif mod == 'dampstiff':
        Bbt = 3e-2
        robot.E = 109e9
    elif mod == 'lengthstiff':
        robot.L = 0.3
        robot.E = 109e9
    else:
        raise Exception('Unknown mod ' + mod)

    if isinstance(robot, CosseratRod):
        robot.Bbt = np.diag([Bbt, Bbt, Bbt])
    else:
        robot.Bbt = torch.diag(torch.tensor([Bbt, Bbt, Bbt], device=robot.device))
    robot.compute_intermediate_terms()

def setup_robot(robot, mod=None, original=False):
    """Set up robot based on expeirmental parameters"""
    if original:
        return setup_robot_original(robot, mod)
    # Measured on the robot
    robot.del_t = 0.05
    robot.L = 0.635 # 25 inches
    robot.tendon_offset = 0.04445 # 1.75 in

    # Information from https://www.mcmaster.com/8576K11/
    robot.r = 0.003175 # 1/4 diameter rod
    robot.rho = 1411.6751 # 0.051 lbs./cu. in.
    robot.E = 2.757903e9 # 400,000 psi

    Bbt = 3e-2 # Damping matrix for delrin

    if mod == None:
        pass
    elif mod == 'noair':
        if isinstance(robot, CosseratRod):
            robot.C = np.array([0, 0, 0])
        else:
            robot.C = torch.tensor([0, 0, 0], device=robot.device) # disable air resistance term
    elif mod == 'nsw':
        if isinstance(robot, CosseratRod):
            robot.g = np.array([0, 0, 0])
        else:
            robot.g = torch.tensor([0, 0, 0], device=robot.device)
    elif mod == 'short':
        robot.L = 0.4
    elif mod == 'damping':
        Bbt = 0.2
    elif mod == 'dampstiff':
        Bbt = 0.2
        robot.E = 10e9
    elif mod == 'lengthstiff':
        robot.L = 0.4
        robot.E = 10e9
    elif mod == 'youngs':
        robot.E = 10e9
    else:
        raise Exception('Unknown mod ' + mod)

    if isinstance(robot, CosseratRod):
        robot.Bbt = np.diag([Bbt, Bbt, Bbt])
    else:
        robot.Bbt = torch.diag(torch.tensor([Bbt, Bbt, Bbt], device=robot.device))
    robot.compute_intermediate_terms()

def simulate(robot, ctl, robot_reference=None):
    if robot_reference is None:
        robot_reference = robot
    y = np.vstack([np.zeros((2, robot_reference.N)), # x and y positions of all elements are zero
                   np.linspace(0, robot_reference.L, robot_reference.N), # z positions of all elements are linearly interpolated
                   np.ones((1, robot_reference.N)), # w of quaternion is 1
                   np.zeros((15, robot_reference.N))]).astype(np.float64) # other state variables of all elements are zero
    z = np.vstack([np.zeros((2, robot_reference.N)),
               np.ones((1, robot_reference.N)),
               np.zeros((3, robot_reference.N))]).astype(np.float64)
    y_prev = y.copy().astype(np.float64)
    z_prev = z.copy().astype(np.float64)
    G = np.zeros(6).astype(np.float64)  # Shooting method initial guess, the net moment and force at the base of the robot
    trajectory = [np.vstack([y, z, y, z]).astype(np.float64)] # List to store robot states, initialized with the initial condition

    for controls in ctl:
        robot.tendon_tensions = np.array(controls).astype(np.float64)

        # Set history terms - Eq(5)
        yh = robot.c1 * y + robot.c2 * y_prev
        zh = robot.c1 * z + robot.c2 * z_prev
        y_prev = y.copy().astype(np.float64)
        z_prev = z.copy().astype(np.float64)

        # Midpoints are linearly interpolated for RK4
        yh_int = 0.5 * (yh[:, :-1] + yh[:, 1:]).astype(np.float64) # length is one less than yh?
        zh_int = 0.5 * (zh[:, :-1] + zh[:, 1:]).astype(np.float64)

        # Shooting method solver call
        # if G is not a numpy array, get x from it, for using minimize() instead of fsolve()
        if not isinstance(G, np.ndarray):
            G = G.x

        if robot.use_fsolve:
            G = fsolve(robot.getResidualEuler, G, args=(y, z, yh, yh_int, zh, zh_int))
        else:
            G = minimize(robot.getResidualEuler,
                        G,
                        args=(y, z, yh, yh_int, zh, zh_int),
                        method='L-BFGS-B')  # this method is faster than the default option
        # concatenate y and z
        traj = np.vstack([y.copy(), z.copy(), yh.copy(), zh.copy()]).astype(np.float64)
        # normalize the quaternions
        # for j in range(robot.N):
            # traj[3:7,j] /= np.linalg.norm(traj[3:7,j])
        trajectory.append(traj)

    return np.array(trajectory)[:-1]

def read_bag(filename):
    b = bagreader(filename)
    print(b.topic_table)
    print()

    base = pd.read_csv(b.message_by_topic('/vicon/continuum_base/pose')).set_index('Time').add_prefix('base.')
    link0 = pd.read_csv(b.message_by_topic('/vicon/continuum_0/pose')).set_index('Time').add_prefix('link0.')
    link1 = pd.read_csv(b.message_by_topic('/vicon/continuum_1/pose')).set_index('Time').add_prefix('link1.')
    link2 = pd.read_csv(b.message_by_topic('/vicon/continuum_2/pose')).set_index('Time').add_prefix('link2.')
    link3 = pd.read_csv(b.message_by_topic('/vicon/continuum_3/pose')).set_index('Time').add_prefix('link3.')
    tension = pd.read_csv(b.message_by_topic('/tension')).set_index('Time').add_prefix('tension.')
    msg = pd.read_csv(b.message_by_topic('/rosout')).set_index('Time')
    tension_cmd = msg['msg'].str.extract(r'Serial Command: (\d+) (\d+) (\d+) (\d+)')

    # Merge everything by time
    merged = pd.merge_ordered(
        pd.merge_ordered(
            pd.merge_ordered(
                pd.merge_ordered(
                    base, link0, on='Time', fill_method="ffill"
                ),
                link1, on='Time', fill_method="ffill"
            ),
            link2, on='Time', fill_method="ffill"
        ),
        link3, on='Time', fill_method="ffill"
    )
    # Filter only rows with non-null values
    # (removes the first few rows where not all locations have been recorded)
    merged = merged[~merged.isnull().any(axis=1)]
    trim = merged.Time.min() + to_trim.get(args.experiment, 0)
    tension.index -= trim
    tension_cmd.index -= trim - 0.06 # Fudge factor to account for time difference in sending command vs receiving tension
    merged.Time -= trim # Start all times at zero

    print('Total time of data', merged.Time.max())

    # plt.plot(tension.index, tension['tension.quaternion.x'])
    # plt.plot(tension.index, tension['tension.quaternion.y'])
    # plt.plot(tension.index, tension['tension.quaternion.z'])
    # plt.plot(tension.index, tension['tension.quaternion.w'])
    # plt.show()

    controls = []
    controls_cmd = []

    # end_t = min(merged.Time.max() / 5, 8)
    end_t = merged.Time.max()
    ts = np.arange(0, end_t, robot.del_t)

    def interpolate_zoh(new_time, original_time, original_values):
        """Interpolate via zero-order hold."""
        zoh_values = np.zeros_like(new_time)

        for i, t in enumerate(new_time):
            # Find the index of the largest time in original_time that is less than or equal to t
            idx = np.searchsorted(original_time, t, side='right') - 1
            if idx < 0:
                idx = 0
            idx = int(idx)
            zoh_values[i] = original_values.iloc[idx] if isinstance(original_values, pd.Series) else original_values[idx]

        return zoh_values

    tensions = np.vstack([
        interpolate_zoh(ts, tension.index, a) for a in [
            tension['tension.quaternion.y'],
            tension['tension.quaternion.z'],
            tension['tension.quaternion.w'],
            tension['tension.quaternion.x'],
        ]]).T

    tensions_cmd = np.vstack([
        interpolate_zoh(ts, tension_cmd.index, a) for a in [
            tension_cmd[1],
            tension_cmd[2],
            tension_cmd[3],
            tension_cmd[0],
        ]]).T

    for t, tens in zip(ts, tensions):
        tendon_tensions = tens.astype(np.float64)
        tendon_tensions = tendon_tensions / 1000 * 9.81
        controls.append(tendon_tensions)

    for t, tens in zip(ts, tensions_cmd):
        tendon_tensions = tens.astype(np.float64)
        tendon_tensions = tendon_tensions / 1000 * 9.81
        controls_cmd.append(tendon_tensions)

    plt.figure(figsize=(6,2))
    plt.xlim(0, 10)
    plt.ylabel('Tension (N)')
    plt.xlabel('Time (s)')

    for i, color in enumerate(['blue', 'orange', 'green', 'magenta']):
        plt.plot(ts, np.array(controls_cmd)[:,i], label=f'T{i+1} Cmd')
        plt.fill_between(ts, np.array(controls)[:,i], np.array(controls_cmd)[:,i], alpha=0.5, label=f'T{i+1} Error')
    plt.legend(ncols=4)
    plt.tight_layout(pad=0)
    plt.show()

    print('Preprocessing')
    positions, orientations, interpolated = preprocessed(merged, ts)

    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(6, 6))

    orientation_quat = []
    for i in range(5):
        orientation_quat.append(Rotation.concatenate(orientations[i]).as_quat(canonical=True))

        # ax = fig.add_subplot(5, 1, i+1)
        # ax.plot(Rotation.concatenate(orientations[i]).as_quat(canonical=True)[:], marker='.')
        # ax.plot(interpolated[:,3:7,[0,3,5,7,9][i]], marker='.')
        # ax.plot(interpolated[:,0:3,[0,3,5,7,9][i]], marker='.', label='t')
        # ax.plot(positions[:, i], marker='.', label='i')
        # ax.legend(loc='best')

    # plt.tight_layout()
    # plt.show()
    print('Preprocessed!')

    controls = np.array(controls).astype(np.float64) # shape is: [n_steps-1, 4]
    return ts, controls, interpolated, positions, orientation_quat

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate KNODE.')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('experiment', type=str, help='path to physical experiment data')
    args = parser.parse_args()

    robot = CosseratRod(use_fsolve=True, nn_path=args.model)
    setup_robot(robot)

    if os.path.exists(f'datas/{os.path.basename(args.experiment)}.npy'):
        data = np.load(f'datas/{os.path.basename(args.experiment)}.npy', allow_pickle=True).item()
        controls = data['controls']
        interpolated = data['interpolated']
        ts = data['t']

        trajectory = simulate(robot, controls)

    elif args.model is None:
        ts, controls, interpolated, pos, ori = read_bag(args.experiment)
        trajectory = simulate(robot, controls)

        np.save(f'datas/{os.path.basename(args.experiment)}.npy', {
            "t": ts,
            "traj": trajectory,
            "controls": controls,
            # "base": positions[0],
            # "link0": positions[1],
            # "link1": positions[2],
            # "link2": positions[3],
            # "link3": positions[4],
            "interpolated": interpolated,
            "positions": pos,
            "orientation": ori,
        })

    else:
        raise Exception('No data file generated yet. Please run without a NN model.')

    plt.title(os.path.basename(args.experiment) + (' (no KNODE)' if args.model is None else ' (with KNODE)'))
    plt.plot(ts, trajectory[:, 0, -1], label='predicted tip X', color='red')
    plt.plot(ts, trajectory[:, 1, -1], label='predicted tip Y', color='green')
    plt.plot(ts, trajectory[:, 2, -1], label='predicted tip Z', color='blue')

    plt.plot(ts, interpolated[:, 0, 9], label='X measured', color='orange')
    plt.plot(ts, interpolated[:, 1, 9], label='Y measured', color='lime')
    plt.plot(ts, interpolated[:, 2, 9], label='Z measured', color='cyan')
    plt.ylabel('Position (m)')
    plt.legend()


    # ax2 = plt.twinx()
    # ax2.plot(ts, controls[:, 0], label='Tension 0', color='black')
    # ax2.plot(ts, controls[:, 1], label='Tension 1', color='darkgrey')
    # ax2.plot(ts, controls[:, 2], label='Tension 2', color='lightgrey')
    # ax2.plot(ts, controls[:, 3], label='Tension 3', color='yellow')
    # plt.ylabel('Tension (N)')
    # plt.legend()

    plt.figure()
    plt.title(os.path.basename(args.experiment) + (' (no KNODE)' if args.model is None else ' (with KNODE)'))
    plt.plot(ts, trajectory[:, 3, -1], label='predicted tip W', color='purple')
    plt.plot(ts, trajectory[:, 4, -1], label='predicted tip X', color='red')
    plt.plot(ts, trajectory[:, 5, -1], label='predicted tip Y', color='green')
    plt.plot(ts, trajectory[:, 6, -1], label='predicted tip Z', color='blue')

    plt.plot(ts, interpolated[:, 3, 9], label='W measured', color='magenta')
    plt.plot(ts, interpolated[:, 4, 9], label='X measured', color='orange')
    plt.plot(ts, interpolated[:, 5, 9], label='Y measured', color='lime')
    plt.plot(ts, interpolated[:, 6, 9], label='Z measured', color='cyan')
    plt.ylabel('Position (m)')
    plt.legend()

    # estimated_state_traj = estimate_state(trajectory[:,:7, :], controls, robot)
    #
    #
    # plt.figure()
    # plt.plot(ts, estimated_state_traj[:, 13:16, 9], color='red', label='Estimated q from physics')
    # plt.plot(ts, estimated_state_meas[:, 13:16, 9], color='orange', label='Estimated q from measurements')
    # plt.plot(ts, trajectory[:, 13:16, 9], color='blue', label='Predicted q')
    # plt.legend()
    plt.show()

    tip_pos = interpolated[:, 0:3, 9]

    print('DTW Distance X', fastdtw(trajectory[:,0,-1], tip_pos[:,0])[0])
    print('DTW Distance Y', fastdtw(trajectory[:,1,-1], tip_pos[:,1])[0])
    print('DTW Distance Z', fastdtw(trajectory[:,2,-1], tip_pos[:,2])[0])
    print('---------------')
    print('DTW Distance XYZ', fastdtw(trajectory[:,:3,-1], tip_pos)[0])
