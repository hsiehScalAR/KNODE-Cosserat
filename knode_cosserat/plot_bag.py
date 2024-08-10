import bagpy
from bagpy import bagreader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation
from preprocess import preprocessed

b = bagreader('physical_experiment_data/dir_a_tension_1400.bag')
print(b.topic_table)

base = pd.read_csv(b.message_by_topic('/vicon/continuum_base/pose')).set_index('Time').add_prefix('base.')
link0 = pd.read_csv(b.message_by_topic('/vicon/continuum_0/pose')).set_index('Time').add_prefix('link0.')
link1 = pd.read_csv(b.message_by_topic('/vicon/continuum_1/pose')).set_index('Time').add_prefix('link1.')
link2 = pd.read_csv(b.message_by_topic('/vicon/continuum_2/pose')).set_index('Time').add_prefix('link2.')
link3 = pd.read_csv(b.message_by_topic('/vicon/continuum_3/pose')).set_index('Time').add_prefix('link3.')

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
merged.Time -= merged.Time.min() # Start all times at zero

print('Total time of data', merged.Time.max())

positions, orientations, interpolated = preprocessed(merged)
print('Preprocessed!')

running = True
def on_close(event):
    global running
    running = False

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=10, azim=-60)
line, = ax.plot3D([], [], [], lw=2)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)

axlines = [[ax.plot3D([], [], [], lw=1, color='red')[0],
            ax.plot3D([], [], [], lw=1, color='green')[0],
            ax.plot3D([], [], [], lw=1, color='blue')[0]] for _ in range(10)]

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

def update(frame):
    t = frame * 20 + 3200

    xyzs = (positions[:, t] - positions[0, t]).T
    xyzs[2] += 0.0635
    line.set_data(xyzs[0], xyzs[1])
    line.set_3d_properties(xyzs[2])

    # Plot only keypoints
    # for i, link in enumerate(['base', 'link0', 'link1', 'link2', 'link3']):
    #     for j in range(3):
    #         pos = positions[i][t] - positions[0, t]
    #         dir = orientations[i][t].as_matrix() @ (0.1 * np.eye(3)[j])
    #         axlines[i][j].set_data([pos[0], pos[0] + dir[0]], [pos[1], pos[1] + dir[1]])
    #         axlines[i][j].set_3d_properties([pos[2], pos[2] + dir[2]])

    # Plot all interpolated points
    for i in range(10):
        for j in range(3):
            pos = interpolated[t, :3, i]
            dir = Rotation.from_quat(interpolated[t, 3:7, i]).as_matrix() @ (0.1 * np.eye(3)[j])
            axlines[i][j].set_data([pos[0], pos[0] + dir[0]], [pos[1], pos[1] + dir[1]])
            axlines[i][j].set_3d_properties([pos[2], pos[2] + dir[2]])
    return [line] + [x for xs in axlines for x in xs]

ani = animation.FuncAnimation(fig, update, frames=500, init_func=init, blit=True, interval=20)
# ani.save('robot-preprocess-tom.gif')

# ani = animation.FuncAnimation(fig, update, frames=len(merged)//20, init_func=init, blit=True, interval=20)

plt.show()
