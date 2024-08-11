import subprocess
import argparse
from tqdm import tqdm
import re
from threading import Thread, Lock
from knode import setup_robot, simulate, CosseratRod
from physics_controls import calc_controls
from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os
import sys

parser = argparse.ArgumentParser(description='Train and Evaluate Multiple Models.')
parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--original', action=argparse.BooleanOptionalAction, default=False, help="use original parameters")
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--fast', action=argparse.BooleanOptionalAction, default=False, help="use fast but inaccurate training")
parser.add_argument('--n_seeds', type=int, default=1)
args = parser.parse_args()

    # 'sine 1.0',
    # 'step 1.0',
    # 'random 0.0',
    # 'sine sine sine 0.5 1.0 2.0',
    # 'sine step step 1.0 1.0 1.0',
    # 'sine sine 0.5 1.0',
    # 'sine sine step 0.5 1.0 1.0',
    # 'sine step 1.0 1.0',
    # 'sine sine random 0.5 1.0 0.0',
    # 'sine random 1.0 0.0',
    # 'sine 0.15',
    # 'sine sine 0.05 0.15',
    # 'sine sine random 0.05 0.15 0.0',
    # 'sine sine sine 0.05 0.15 0.225',
if args.original:
    datas = [
        'sine sine 0.05 0.15',
        'sine sine random 0.05 0.15 0.0',
    ]
else:
    datas = [
        'sine sine 0.5 1.0',
        'sine sine random 0.5 1.0 0.0',
    ]

    # 'sine 2.0',
    # 'step 2.5',
    # 'random 1.0',
    # 'random 1.0'


    # 'sine 0.2',
    # 'step 1.5',
if args.original:
    eval_set = [
        'sine 0.2',
        'step 1.5',
    ]
else:
    eval_set = [
        'sine 1.5',
        'step 1.5',
    ]


mods = [
    # None,
    'nsw',
    'short',
    # 'damping',
    'youngs',
    'lengthstiff',
    # 'dampstiff',
]

lock = Lock()

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

class Train(Thread):
    """Thread to launch and monitor the training process for a dataset."""

    def __init__(self, data, mod, i, seed):
        super().__init__()
        self.progress = tqdm(total=args.epochs, position=i)
        self.opts = ("" if mod is None else mod).ljust(5)
        self.progress.set_description(f'{data};{self.opts}f;{seed}')
        self.data = data
        self.mod = mod
        self.seed = seed

    def run(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'physics_train.py')
        cmd = [sys.executable, '-u', filename, '--verbose', '--no-eval', '--epochs', str(args.epochs), '--seed', str(self.seed)]
        if self.mod is not None:
            cmd.extend(['--mod', self.mod])
        if args.original:
            cmd.append('--original')
        if args.fast:
            cmd.append('--fast')
        control_type, control_arg = split_list(self.data.split(' '))
        cmd.extend([*control_type, *control_arg])
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.progress.reset()
        for line in proc.stdout:
            line = line.decode('utf-8')
            epoch = re.search(r'Epoch (\d+)', line)
            loss = re.search(r'Total loss: (.*?), lr (.*?)', line)

            if epoch is not None:
                with lock:
                    self.progress.update(10) # Progress counts in steps of 10
            if loss is not None:
                with lock:
                    self.progress.set_description(f'{self.data};{self.opts};{self.seed} Loss: {float(loss.group(1)):.6f}')

def batch(iterable, n=1):
    """Batch an array into chuncks of n items"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

robot_reference = CosseratRod(use_fsolve=True)
setup_robot(robot_reference, original=args.original)

def calc_evaldata(control_type, control_arg):
    controls = np.array(calc_controls(control_type, float(control_arg), robot_reference.del_t, 100))
    reference_traj = simulate(robot_reference, controls)[:, :25]
    return {
        'controls': controls,
        'interpolated': reference_traj
    }

if args.train:
    threads = []
    # Create the threads
    count = 0
    for i, data in enumerate(datas):
        for j, mod in enumerate(mods):
            for s in range(args.n_seeds):
                t = Train(data, mod, count, s)
                t.daemon = True
                threads.append(t)
                count += 1

    # Start the threads in batches of 2. Helps keep system utilization low.
    for bat in batch(threads, 2):
        for t in bat:
            t.start()
        for t in bat:
            t.join()

    # Separate the evaluation results
    for _ in range(40):
        print()

def pct_error(new, old):
    if old == 0:
        return 0 if new == 0 else float('inf')
    return (new - old) / old * 100

SPACE = 40
if args.eval:
    eval_np = [calc_evaldata(*evall.split(' ')) for evall in eval_set]

    print(' ' * SPACE, end='')
    for e in eval_set:
        print((';' + e + ' DTW').ljust(20), end='')
        print((';' + e + ' PQ MSE').ljust(20), end='')
    print()

    prefix = 'physics_original' if args.original else 'physics'
    baselines = {}
    for data in [None, *datas]:
        for mod in mods:
            for seed in range(args.n_seeds):
                if data is None:
                    data_short = f'baseline {mod}'
                    robot = CosseratRod(use_fsolve=True, nn_path=None)
                    setup_robot(robot, mod, args.original)
                else:
                    data_short = f'{data} {mod} {seed}'
                    filename = '_'.join('-'.join(s).replace('.', '_') for s in split_list(data.split(' ')))
                    nn_path = f'saved_models/{prefix}_{filename}_{mod}_trainlen_30_{args.epochs}_epoch_{seed}.pth'
                    robot = CosseratRod(use_fsolve=True, nn_path=nn_path)
                    setup_robot(robot, mod, args.original)

                print(data_short.ljust(SPACE), end='')

                for evall, eval_data in zip(eval_set, eval_np):
                    tensions = eval_data['controls']
                    tip_pos = eval_data['interpolated'][:, :3, 9]
                    trajectory = simulate(robot, tensions)
                    filename = evall.replace(' ', '_') + '+' + data_short.replace(' ', '_')
                    np.save(f'evals/{prefix}_{filename}_trainlen_30_{args.epochs}_epochs.npy', {
                        "tensions": eval_data['controls'],
                        "reference": eval_data['interpolated'],
                        "predicted": trajectory
                    })
                    # plt.plot(trajectory[:,:3,9], label='trajectory')
                    # plt.plot(tip_pos, label='reference')
                    # plt.legend()
                    # plt.draw()
                    # plt.pause(0.001)
                    # plt.cla()
                    # plt.show()
                    dtw = fastdtw(trajectory[:,:3,9], tip_pos)[0]

                    se_pos = (trajectory[:, :3] - eval_data['interpolated'][:, :3]).reshape((-1, 3)) ** 2
                    eval_quat = trajectory[:, 3:7].transpose((0, 2, 1)).reshape((-1, 4))
                    ref_quat =  eval_data['interpolated'][:, 3:7].transpose((0, 2, 1)).reshape((-1, 4))
                    eval_euler = Rotation.from_quat(eval_quat, scalar_first=True).as_euler('zyx')
                    ref_euler = Rotation.from_quat(ref_quat, scalar_first=True).as_euler('zyx')
                    se_euler = (eval_euler - ref_euler) ** 2

                    mse = np.mean(np.concatenate([se_euler, se_pos])) * 1000
                    if data is None:
                        baselines[(evall,mod)] = { 'dtw': dtw, 'mse': mse }
                        print(';{0:.2f}'.format(dtw).ljust(20), end='')
                        print(';{0:.2f}'.format(mse).ljust(20), end='')
                    else:
                        base = baselines[(evall,mod)]
                        change = pct_error(dtw, base['dtw'])
                        print(f';{dtw:.2f} ({change:+.1f}%)'.ljust(20), end='')
                        change = pct_error(mse, base['mse'])
                        print(f';{mse:.2f} ({change:+.1f}%)'.ljust(20), end='')
                print()
