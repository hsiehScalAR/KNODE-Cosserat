import numpy as np
import matplotlib.pyplot as plt
import argparse
from fastdtw import fastdtw
from scipy.spatial.transform import Rotation
import torch
from knode import *

parser = argparse.ArgumentParser(description='Plot KNODE.')
parser.add_argument('--original', action=argparse.BooleanOptionalAction, default=False, help="use original parameters")
parser.add_argument('--n_seeds', type=int, default=5)

plt.rcParams['svg.fonttype'] = 'none'

args = parser.parse_args()

    # # 'sine 2.0',
    # 'sine 1.0',
    # # 'step 2.5',
    # 'step 1.0',
    # 'random 0.0',
    # # 'random 1.0'
    # 'sine sine sine 0.5 1.0 2.0',

    # 'sine step random 1.0 1.0 0.0',
    # 'sine step step 1.0 1.0 1.5',
    # 'sine sine 0.5 1.0',
    # 'sine sine step 0.5 1.0 1.5',
    # 'sine step 1.0 1.0'

if args.original:
    datas = [
        'sine sine 0.05 0.15',
        'sine sine random 0.05 0.15 0.0',
    ]
else:
    datas = [
        # 'sine 1.0',
        'sine sine 0.5 1.0',
        # 'sine sine random 0.5 1.0 0.0',
    ]

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
    'nsw',
    'short',
    # 'damping',
    'youngs',
    'lengthstiff',
    # 'dampstiff',
]

# for evall in eval_set:
#     labels = []
#     results = []
#     for data in ['baseline', *datas]:
#         for mod in mods:
#             filename = evall.replace(' ', '_') + '+' + data.replace(' ', '_')
#             filename = f'evals/physics_{filename}_{mod}_trainlen_30_1000_epochs.npy'
#             result = np.load(filename, allow_pickle=True)
#             labels.append(f'{data} {mod}')
#             results.append(result.item())
#     plt.title(f'Model generalization to {evall} Trajectory: X axis')

#     for lab, res in zip(labels, results):
#         alpha = 0.5
#         if 'baseline' in lab:
#             color = 'red'
#             alpha = 1
#         elif 'sine' in lab:
#             color = 'green'
#         elif 'random' in lab:
#             color = 'blue'
#         elif 'step' in lab:
#             color = 'magenta'
#         else:
#             color = ''
#         plt.plot(res['predicted'][:,0,9], color, label=lab, alpha=alpha)

#     plt.plot(results[0]['reference'][:,0,9], 'k^-', label='Reference')
#     plt.legend(fontsize=8)
#     plt.show()

def pct_error(new, old):
    if old == 0:
        return 0 if new == 0 else float('inf')
    return (new - old) / old * 100

SPACE = 40
baselines = {}

print(' ' * SPACE, end='')
for e in eval_set:
    print((';' + e + ' DTW').ljust(20), end='')
    print((';' + e + ' PQ MSE').ljust(20), end='')
print()

for data in ['baseline', *datas]:
    for mod in mods:
        if data == 'baseline':
            data_short = f'baseline {mod}'
        else:
            data_short = f'{data} {mod}'
        print(data_short.ljust(SPACE), end='')

        for evall in eval_set:
            results = []
            for i in range(args.n_seeds):
                filename = evall.replace(' ', '_') + '+' + data.replace(' ', '_') + '_' + mod
                if data != 'baseline':
                    filename += f'_{i}'
                prefix = 'physics_original' if args.original else 'physics'
                filename = f'evals/{prefix}_{filename}_trainlen_30_1000_epochs.npy'
                results.append(np.load(filename, allow_pickle=True).item())

            dtws = [fastdtw(d['predicted'][:,:3,9], d['reference'][:,:3,9])[0] for d in results]
            mses = []
            for d in results:
                se_pos = (d['predicted'][:, :3] - d['reference'][:, :3]).reshape((-1, 3)) ** 2
                eval_quat = d['predicted'][:, 3:7].transpose((0, 2, 1)).reshape((-1, 4))
                ref_quat =  d['reference'][:, 3:7].transpose((0, 2, 1)).reshape((-1, 4))
                eval_euler = Rotation.from_quat(eval_quat, scalar_first=True).as_euler('zyx')
                ref_euler = Rotation.from_quat(ref_quat, scalar_first=True).as_euler('zyx')
                se_euler = (eval_euler - ref_euler) ** 2
                mses.append(np.mean(np.concatenate([se_euler, se_pos])) * 1000)
            dtw = sum(dtws) / len(dtws)
            mse = sum(mses) / len(mses)
            if data == 'baseline':
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

def split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]

i = 0
for data in datas:
    for mod in mods:
        i += 1
        ax = plt.subplot(2, 2, i)
        print(ax)
        plt.title(f'Loss for {data} {mod}')
        losses = []
        for seed in range(args.n_seeds):
            filename = '_'.join('-'.join(s).replace('.', '_') for s in split_list(data.split(' ')))
            nn_path = f'saved_models/{prefix}_{filename}_{mod}_trainlen_30_1000_epoch_{seed}.pth'
            results = torch.load(nn_path)
            # plt.semilogy(results['loss'][0:], label=f'seed {seed}')
            losses.append(results['loss'])
        epochs = np.arange(len(losses[0]))[10:]
        losses = np.array(losses)[:, 10:]
        average = np.mean(losses , axis=0)
        median = np.median(losses, axis=0)
        lower_perc = np.min(losses, axis=0)
        upper_perc = np.max(losses, axis=0)

        # ax.set_yscale('log')
        ax.plot(epochs, average, label='Loss mean')
        ax.set_xlabel('Epochs')
        # plt.semilogy(lower_perc, label='25th percentile')
        # plt.semilogy(upper_perc, label='75th percentile')
        ax.fill_between(epochs, lower_perc, upper_perc, alpha=0.3, label='Loss range')
        plt.legend()
plt.tight_layout()
plt.show()


for evall in eval_set:
    plt.figure(figsize=(12, 4))
    plt.suptitle(f'Model generalization to {evall} Trajectory: Tip X axis')
    for i, mod in enumerate(mods):
        plt.subplot(2, 2, i+1)

        plt.title(mod)
        for data in [*datas, 'baseline']:
            filename = evall.replace(' ', '_') + '+' + (data + ' ' + mod).replace(' ', '_')
            prefix = 'physics_original' if args.original else 'physics'
            if data != 'baseline':
                filename += f'_{i}'
            filename = f'evals/{prefix}_{filename}_trainlen_30_1000_epochs.npy'
            result = np.load(filename, allow_pickle=True).item()
            ts = np.arange(result['tensions'].shape[0]) * 0.05

            lab = data
            linestyle = 'solid'
            marker = ''
            if 'baseline' in lab:
                color = 'red'
                alpha = 1
                # marker='|'
            elif 'sine random' in lab:
                color = 'blue'
            elif 'sine sine' in lab:
                color = 'green'
            elif 'random' in lab:
                color = 'cyan'
                linestyle = 'dashed'
            elif 'sine' in lab:
                color = 'lime'
                linestyle = 'dashed'
            elif 'step' in lab:
                color = 'pink'
                linestyle = 'dashed'
            else:
                color = ''
            plt.plot(ts, result['predicted'][:,0,9], color, linestyle=linestyle, label=lab, marker=marker)

        plt.plot(ts, result['reference'][:,0,9], 'k-', label='Reference')
        plt.legend(loc='upper right', ncol=3)
        plt.xlabel('Time (s)')
        plt.ylabel('Tip Position X (m)')
    plt.tight_layout(pad=0.5)
    plt.show()
