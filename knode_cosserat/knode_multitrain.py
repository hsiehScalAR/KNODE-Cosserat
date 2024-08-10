import subprocess
import argparse
from tqdm import tqdm
import re
from threading import Thread, Lock
from knode import setup_robot, simulate, CosseratRod
from fastdtw import fastdtw
import numpy as np

parser = argparse.ArgumentParser(description='Train and Evaluate Multiple Models.')
parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eval', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

datas = [
    'datas/sin_1_0_amp_300.bag.npy',
    'datas/sin_3_0_amp_300.bag.npy',
    'datas/dir_a_tension_800.bag.npy',
    'datas/dir_a_tension_950.bag.npy',
    'datas/rand_0_60s.bag.npy',
    'datas/rand_1_60s.bag.npy',
]

eval_set = [
    'datas/sin_1_0_amp_300.bag.npy',
    'datas/sin_3_0_amp_300.bag.npy',
    'datas/dir_a_tension_800.bag.npy',
    'datas/dir_a_tension_950.bag.npy',
    'datas/rand_0_60s.bag.npy',
    'datas/rand_1_60s.bag.npy',
]

lock = Lock()

class Train(Thread):
    """Thread to launch and monitor the training process for a dataset."""

    def __init__(self, data, air, i):
        super().__init__()
        self.progress = tqdm(total=4000, position=i)
        self.progress.set_description(f'{data};{"     " if air else "noair"}')
        self.data = data
        self.air = air

    def run(self):
        cmd = ['python', '-u', 'knode_train.py', '--verbose', '--air' if self.air else '--no-air', self.data]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for line in proc.stdout:
            line = line.decode('utf-8')
            epoch = re.search(r'Epoch (\d+)', line)
            loss = re.search(r'Total loss: (.*?), lr (.*?)', line)

            if epoch is not None:
                with lock:
                    self.progress.update(10) # Progress counts in steps of 10
            if loss is not None:
                with lock:
                    opts = "     " if self.air else "noair"
                    self.progress.set_description(f'{self.data};{opts} Loss: {float(loss.group(1)):.6f}')

def batch(iterable, n=1):
    """Batch an array into chuncks of n items"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

if args.train:
    threads = []
    # Create the threads
    for i, data in enumerate(datas):
        for j, air in enumerate([True, False]):
            t = Train(data, air, 2*i + j)
            t.daemon = True
            threads.append(t)

    # Start the threads in batches of 2. Helps keep system utilization low.
    for bat in batch(threads, 2):
        for t in bat:
            t.start()
        for t in bat:
            t.join()

    # Separate the evaluation results
    for _ in range(6):
        print()

if args.eval:
    eval_np = [np.load(evall, allow_pickle=True).item() for evall in eval_set]

    print(' ' * 20, end='')
    for e in eval_set:
        print(e.split('/')[1].replace('.bag.npy', '').ljust(20), end='')
    print()

    for data in [None, *datas]:
        for air in [True, False]:
            if data is None:
                data_short = 'baseline'
                robot = CosseratRod(use_fsolve=True, nn_path=None)
                setup_robot(robot, air)
            else:
                data_short = '-'.join(s.split('/')[-1].replace('.bag.npy', '').replace('_1000_steps_10_grid', '') for s in [data])
                nn_path = f'saved_models/{data_short}+{air}_trainlen_150_4000_epoch.pth'
                robot = CosseratRod(use_fsolve=True, nn_path=nn_path)
                setup_robot(robot, air)

            if not air:
                data_short += 'nsw'
            print(data_short.ljust(20), end='')

            for evall, eval_data in zip(eval_set, eval_np):
                tensions = eval_data['controls']
                tip_pos = eval_data['interpolated'][:, :3, 9]
                trajectory = simulate(robot, tensions)
                print('{0:.2f}'.format(fastdtw(trajectory[:,:3,-1], tip_pos)[0]).ljust(20), end='')
            print()
