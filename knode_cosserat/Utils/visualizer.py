import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
mpl.rcParams['legend.fontsize'] = 20



def visualize_2d(robot, y, i):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(y[2,:], y[0,:])
    ax.set_title('CantileverRod')
    ax.set_xlabel('z(m)')
    ax.set_ylabel('x(m)')
    ax.axis([0, 1.1*robot.L, -0.55*robot.L, 0.55*robot.L])

    # Plot the step number on the plot
    step_text = f"Step: {i}"
    ax.text(0.05*robot.L, 0.5*robot.L, step_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    filename = f"images/frame_{i:04d}.png"
    plt.savefig(filename)
    plt.close(fig)  # Close the current figure to free up memory
    return filename  # Return filename for convenience

class ContinuumRobotVisualizer:

    def __init__(self, results, robot):
        self.results = results
        self.robot = robot

    def plot_robot_3d(self, ax, timestep):
        ax.cla()  # Clear the previous frame
        
        # Plotting the centerline
        x = self.results[timestep, 0, :]
        y = self.results[timestep, 1, :]
        z = self.results[timestep, 2, :]
        ax.plot(x, y, z, label='Centerline', color='b')
        # Plot the step number on the plot
        step_text = f"Step: {timestep}"
        ax.text(0.05*self.robot.L, 0.5*self.robot.L, self.robot.L, step_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        # Plotting the tendons, assuming the tendons are equally spaced in terms of angle
        for i in range(self.robot.n_tendons):
            angle = 2 * np.pi * i / self.robot.n_tendons
            tendon_x = x + self.robot.tendon_offset * np.cos(angle)
            tendon_y = y + self.robot.tendon_offset * np.sin(angle)
            ax.plot(tendon_x, tendon_y, z, linestyle='--', color='r')

        ax.set_xlim([np.min(self.results[:, 0, :]) - 0.1, np.max(self.results[:, 0, :]) + 0.1])
        ax.set_ylim([np.min(self.results[:, 1, :]) - 0.1, np.max(self.results[:, 1, :]) + 0.1])
        ax.set_zlim([np.min(self.results[:, 2, :]) - 0.1, np.max(self.results[:, 2, :]) + 0.1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    def plot_robot_2d(self, nominal, ref, timestep):        
        # Plotting the centerline
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(121)
        ax2 = fig1.add_subplot(122)
        x = self.results[timestep, 0, :]
        y = self.results[timestep, 1, :]
        z = self.results[timestep, 2, :]
        
        x_ref = ref[timestep, 0, :]
        y_ref = ref[timestep, 1, :]
        z_ref = ref[timestep, 2, :]

        x_nom = nominal[timestep, 0, :]
        y_nom = nominal[timestep, 1, :]
        z_nom = nominal[timestep, 2, :]

        ax1.set_title('X-Z view - Step:' + str(timestep))
        ax1.plot(x, z, label='predicted', color='b', marker='.')
        ax1.plot(x_ref, z_ref, label='ref', color='orange', marker='.')
        ax1.plot(x_nom, z_nom, label='nom', color='g', marker='.')
        ax2.set_title('Y-Z view - Step:' + str(timestep))
        ax2.plot(y, z, label='predicted', color='b', marker='.')
        ax2.plot(y_ref, z_ref, label='ref', color='orange', marker='.')
        ax2.plot(y_nom, z_nom, label='nom', color='g', marker='.')

        ax1.set_xlim([np.min(self.results[:, 0, :]) - 0.1, np.max(self.results[:, 0, :]) + 0.1])
        ax1.set_ylim([np.min(self.results[:, 2, :]) - 0.1, np.max(self.results[:, 2, :]) + 0.1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')

        ax2.set_xlim([np.min(self.results[:, 1, :]) - 0.1, np.max(self.results[:, 1, :]) + 0.1])
        ax2.set_ylim([np.min(self.results[:, 2, :]) - 0.1, np.max(self.results[:, 2, :]) + 0.1])
        ax2.set_xlabel('Y')
        ax2.set_ylabel('Z')
        ax1.legend()
        ax2.legend()

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        def update(num):
            self.plot_robot_3d(ax, num)
            return ax,

        ani = FuncAnimation(fig, update, frames=len(self.results), repeat=False)
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_zlim(-0.1, 0.5)
        plt.show()

    def save_as_gif(self, filename="animations/robot_motion_3d.gif"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ani = FuncAnimation(fig, lambda num: self.plot_robot_3d(ax, num), frames=len(self.results), repeat=False)

        # Save as GIF
        writer = PillowWriter(fps=5)  # Adjust fps for desired speed
        ani.save(filename, writer=writer)

def plot_2d(data_arr, data_arr2=None, legend=None, markers=None,title="rod tip position"):
    """
    Plots an array of data using matplotlib.
    
    Args:
    - data (list or np.ndarray): The array of data to plot.
    """
    # plot the tip of the rod
    fig = plt.figure(figsize=(15,12))
    ax1 = fig.add_subplot(3,2,1)
    ax2 = fig.add_subplot(3,2,3)
    ax3 = fig.add_subplot(3,2,5)
    axes = [ax1, ax2, ax3]
    axes_labels = ['x[m]', 'y[m]', 'z[m]']
    ax1.set_title(title + " - tip")
    for i in range(len(data_arr)):
        for idx, ax_idx in enumerate([0, 1, 2]):
            if data_arr2 is not None:
                axes[idx].plot(data_arr[i][:, ax_idx, 5], data_arr2[i][:, ax_idx,5], label=legend[i, idx],marker=markers[i])
            else:
                axes[idx].plot(data_arr[i][:, ax_idx, 5], label=legend[i], marker=markers[i])
            axes[idx].grid(True)
            axes[idx].set_xlabel('t')
            axes[idx].set_ylabel(axes_labels[idx])
    
    # plot the root of the rod [2]
    ax4 = fig.add_subplot(3,2,2)
    ax5 = fig.add_subplot(3,2,4)
    ax6 = fig.add_subplot(3,2,6)
    axes2 = [ax4, ax5, ax6]
    axes2_labels = ['x[m]', 'y[m]', 'z[m]']
    ax4.set_title(title + " - root")
    for i in range(len(data_arr)):
        for idx, ax_idx in enumerate([0, 1, 2]):
            if data_arr2 is not None:
                axes2[idx].plot(data_arr[i][:, ax_idx, 2], data_arr2[i][:, ax_idx, 2], label=legend[i, idx],marker=markers[i])
            else:
                axes2[idx].plot(data_arr[i][:, ax_idx, 2], label=legend[i], marker=markers[i])
            axes2[idx].grid(True)
            axes2[idx].set_xlabel('t')
            axes2[idx].set_ylabel(axes2_labels[idx])

    plt.legend()
    
    plt.savefig(title+'.png', format='png', dpi=600, bbox_inches ='tight', pad_inches = 0.1)
    plt.savefig(title+'.eps', format='eps', dpi=600, bbox_inches ='tight', pad_inches = 0)

def compute_traj_MSE(traj1, traj2):
    """
    Computes the mean squared error between two trajectories.
    
    Args:
    - traj1 (np.ndarray): The first trajectory.
    - traj2 (np.ndarray): The second trajectory.
    
    Returns:
    - mse (float): The mean squared error between the two trajectories.
    """
    return np.mean((traj1 - traj2) ** 2)

if __name__ == "__main__":
    step_E209 = np.load('data/test_traj_2000_steps_10_grid_E209.npy', allow_pickle=True).item()["traj"]
    step_E109 = np.load('data/random_traj_2000_steps_10_grid_E209.npy', allow_pickle=True).item()["traj"]

    index = 199

    plot_2d([step_E209[:index],
            step_E109[:index]
    ], 
                legend=["Step E209e9 float32", 
                        "Step E209e9 float64"], 
                markers=["x", "."],
                title="step input")

    step_diff = compute_traj_MSE(step_E209[:index], step_E109[:index])
        
    print(f"step data E209 and E109 difference:\n {step_diff}")
    plt.show()