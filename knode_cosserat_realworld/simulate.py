import numpy as np
import argparse
import os
import torch
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize
from PIL import Image
from Utils.visualizer import ContinuumRobotVisualizer, visualize_2d, plot_2d, compute_traj_MSE
from Utils.data_processing import normalize_data
from cosserat_ode import CosseratRod

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--steps', type=int, help='an integer number', default=100)
    parser.add_argument('--model', type=str, help='a string', default=None)
    parser.add_argument('--save_name', type=str, help='a string', default="quick_test")
    parser.add_argument('--real_data_path', type=str, help='a string', default='data/real_physical/sin_1_0_amp_300_estimated.npy')
    args = parser.parse_args()

    real_data = np.load(args.real_data_path, allow_pickle=True).item()
    real_controls = real_data['controls']
    real_traj = real_data['traj']

    STEPS = args.steps # Number of steps to simulate
    VIS_FRAME = 0  # which frame to visualize in 2D
    SHOW_ANIMATION = True
    save_file_name = args.save_name
    # initialize the robot
    nn_path = args.model

    robot = CosseratRod(use_fsolve=True,
                        nn_path=nn_path)

    clear_imgs = True
    # Initialize to straight configuration
    # y is general ODE state vector
    y = np.vstack([np.zeros((2, robot.N)), # x and y positions of all elements are zero
                np.linspace(0, robot.L, robot.N), # z positions of all elements are linearly interpolated
                np.ones((1, robot.N)), # w of quaternions is 1
                np.zeros((15, robot.N))]).astype(np.float64) # other state variables of all elements are zero

    # z is general vector with relevant time derivatives, but elements not in y
    z = np.vstack([np.zeros((2, robot.N)),
                np.ones((1, robot.N)),
                np.zeros((3, robot.N))]).astype(np.float64)

    # y = real_traj[0, 0:19]
    # z = real_traj[0, 19:25]

    # using initial condition of the data as the simulation initial condition
    #y = real_data['traj'][0, :19, :].copy().astype(np.float64)
    #z = real_data['traj'][0, 19:25, :].copy().astype(np.float64)

    y_prev = y.copy().astype(np.float64)
    z_prev = z.copy().astype(np.float64)

    # Main Simulation Loop - Section 2.4
    G = np.zeros(6).astype(np.float64)  # Shooting method initial guess, the net moment and force at the base of the robot
    frames = []  # List to store frames
    trajectory = [np.vstack([y, z, y, z]).astype(np.float64)] # List to store robot states, initialized with the initial condition
    controls = []

    for i in range(1, STEPS):
        print(f"Step {i} of {STEPS}")
        T1 = real_controls[i, 0]
        T2 = real_controls[i, 1]
        T3 = real_controls[i, 2]
        T4 = real_controls[i, 3]

        tendon_tensions = np.array([T1, T2, T3, T4]).astype(np.float64)  # Control inputs
        robot.tendon_tensions = tendon_tensions
        controls.append(tendon_tensions)

        # Set history terms - Eq(5)
        yh = robot.c1 * y + robot.c2 * y_prev
        zh = robot.c1 * z + robot.c2 * z_prev
        y_prev = y.copy().astype(np.float64)
        z_prev = z.copy().astype(np.float64)

        # Shooting method solver call
        # if G is not a numpy array, get x from it, for using minimize() instead of fsolve()
        if not isinstance(G, np.ndarray):
            G = G.x

        G = fsolve(robot.getResidualEuler, G, args=(y, z, yh, zh))

        # concatenate y and z
        traj = np.vstack([y.copy(), z.copy(), yh.copy(), zh.copy()]).astype(np.float64)
        trajectory.append(traj)
        # frame_filename = visualize_2d(robot, y, i)
        # frames.append(frame_filename)

    trajectory = np.array(trajectory).astype(np.float64) # shape is: [n_steps, 25, N]
    controls = np.array(controls).astype(np.float64) # shape is: [n_steps-1, 4]

    # Save the trajectory
    if not os.path.exists("data"):
        os.makedirs("data")
    np.save('data/' + save_file_name + '.npy',
            {"traj": trajectory, "controls": controls})

    # Create and save an animated gif of the 2D simulation
    if not os.path.exists("animations"):
        os.makedirs("animations")
    # images = [Image.open(frame) for frame in frames]
    # images[0].save('animations/' + save_file_name + '_2d.gif',
                    # save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

    # Optionally remove the individual frames
    # if clear_imgs:
        # for frame in frames:
            # os.remove(frame)

    # Create and save an animated gif of the 3D simulation
    visualizer = ContinuumRobotVisualizer(trajectory, robot)
    # visualizer.save_as_gif(filename="animations/" + save_file_name + "_3d.gif")
    # if SHOW_ANIMATION:
        # visualizer.visualize()

    # Plot the loss curve
    if nn_path is not None:
        loss = torch.load(nn_path, map_location=torch.device('cpu'))["loss"][50:] # truncate the first 50 epochs
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(loss)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")

    fig_c = plt.figure()
    ax1 = fig_c.add_subplot(111)
    ax1.plot(controls[:, 0], label="T1", marker=".")
    ax1.plot(controls[:, 1], label="T2", marker=".")
    ax1.plot(controls[:, 2], label="T3", marker=".")
    ax1.plot(controls[:, 3], label="T4", marker=".")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Tendon Tension")
    ax1.set_title("Control Inputs")
    fig_c.legend()

    ######################################################################
    ########################## Evaluation ################################
    ######################################################################

    test_traj = np.load('data/quick_test.npy', allow_pickle=True).item()["traj"][:,:25,:]
    index = len(test_traj)
    step_E209 = np.load('data/real_physical/sin_1_0_amp_300_estimated.npy', allow_pickle=True).item()["traj"][:,:25,:]
    step_E109 = np.load('data/sim_sin_1_0_amp_300.npy', allow_pickle=True).item()["traj"][:,:25,:]

    plot_2d([
            step_E209[:index],
            step_E109[:index],
            test_traj[:index]
],
            legend=[
                    "True data with hardware",
                    "Simulated data with true params",
                    "KNODE pred"],
            markers=[".", ".", "x"],
            title="step input")

    step_diff = compute_traj_MSE(step_E209[:index], step_E109[:index])
    test_step_diff = compute_traj_MSE(step_E209[:index], test_traj[:index])

    print(f"step data E209 and E109 difference:\n {step_diff}")
    print(f"step data E209 and NNpred difference:\n {test_step_diff}")

    visualizer.plot_robot_2d(step_E109, step_E209, VIS_FRAME)
    plt.show()
