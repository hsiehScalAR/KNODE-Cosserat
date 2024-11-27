import numpy as np
from scipy.linalg import logm, expm
from sklearn.metrics import mean_squared_error
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from interpolate_curve import fit_curve
import argparse

def compute_R_spatial_derivative(R_matrices, arc_lengths):
    """
    Compute the spatial derivative of rotation matrices with respect to arc length.

    Parameters:
    - R_matrices: array of shape [N, 3, 3], rotation matrices along the arc length
    - arc_lengths: array of shape [N], arc lengths

    Returns:
    - R_derivatives: array of shape [N, 3, 3], spatial derivatives of the rotation matrices with respect to arc length
    """
    N = len(arc_lengths)
    R_derivatives = np.zeros((N, 3, 3))

    for i in range(N - 1):
        R_current = R_matrices[i]
        R_next = R_matrices[i + 1]

        # Compute relative rotation
        R_rel = R_next @ R_current.T

        # Compute the logarithm of the relative rotation to get the rotation vector
        log_R_rel = logm(R_rel)

        # Compute the derivative (approximate) with respect to arc length
        delta_s = arc_lengths[i + 1] - arc_lengths[i]
        angular_velocity = log_R_rel / delta_s

        # Ensure the angular velocity is skew-symmetric
        R_derivatives[i] = R_current @ angular_velocity

    # For the last derivative, we assume it's the same as the second to last
    R_derivatives[-1] = R_derivatives[-2]

    return R_derivatives

def compute_v_u(global_positions, quaternions, arc_lengths):
    """
    Compute the linear rate of change of position with respect to arc length at one time step.

    Parameters:
    - global_positions: array of shape [3, N], positions in the global frame
    - quaternions: array of shape [3, N], quaternions in the global frame
    - arc_lengths: array of shape [N], arc lengths

    Returns:
    - rate_of_change: array of shape [3, N-1], rate of change with respect to arc length
    """
    N = len(arc_lengths)
    # Compute rotation matrices from quaternions
    rotation_matrices = np.zeros([N, 3, 3])  # Shape: (N, 3, 3)

    # compute the rate of change of global positions with respect to arc length
    p_s = np.zeros([3, N])
    for i in range(N-1):
      del_s = arc_lengths[i+1] - arc_lengths[i]
      p_s[:, i] = (global_positions[:,i+1] - global_positions[:,i])/del_s
    p_s[:,-1] = p_s[:,-2]

    for i in range(N):
        # Quaternion to Rotation
        h = quaternions[:, i]
        h1, h2, h3, h4 = h
        rotation_matrices[i, :, :] = np.eye(3) + 2 / np.dot(h, h) * \
                      np.array([[-h3**2-h4**2, h2*h3-h4*h1, h2*h4+h3*h1],
                      [h2*h3+h4*h1, -h2**2-h4**2, h3*h4-h2*h1],
                      [h2*h4-h3*h1, h3*h4+h2*h1, -h2**2-h3**2]])

    R_s = compute_R_spatial_derivative(rotation_matrices, arc_lengths)
    v = np.zeros_like(global_positions)
    u = np.zeros_like(global_positions)
    for i in range(N):
        v[:, i] = rotation_matrices[i].T @ p_s[:, i]
        #print("ground truth ps", rotation_matrices[i] @ trajectory[5,19:22,i])
        u_hat = rotation_matrices[i].T @ R_s[i]
        u[0, i] = u_hat[2, 1]
        u[1, i] = u_hat[0, 2]
        u[2, i] = u_hat[1, 0]

    # hard code the base's linear rate of change
    v[0:2, 0] = 0
    v[2, 0] = 1
    return v, u

def pairwise_angular_velocities(q1, q2, dt):
    """
    Compute the angular velocity between a pair of quaternions
    """
    # https://mariogc.com/post/angular-velocity-quaternions/
    return (2 / dt) * np.array([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

def compute_angular_velocities(quaternions, del_t):
    """
    Compute the angular velocities from a series of quaternions
    """
    T, _, N = quaternions.shape
    angular_velocities = np.zeros((T, 3, N))

    for n in range(N):
        for t in range(T-1):
            q1 = quaternions[t, :, n]
            q2 = quaternions[t+1, :, n]

            angular_velocities[t+1, :, n] = pairwise_angular_velocities(q1, q2, del_t)

        # Handle the first time step by setting it to the same as the next step
        angular_velocities[0, :, n] = angular_velocities[1, :, n]

    return angular_velocities


def compute_internal_forces_and_moments(p, arc_lengths, Rs, q, w, qt, wt, tensions, robot):
    """
    Compute n, m
    """

    # estimating based on statics
    n = np.zeros([3, robot.N])
    m = np.zeros([3, robot.N])
    tendon_forces = np.dot(tensions, robot.tendon_dirs)

    # compute the rate of change of global positions with respect to arc length
    p_s = np.zeros([3, robot.N])
    for i in range(robot.N-1):
      del_s = arc_lengths[i+1] - arc_lengths[i]
      p_s[:, i] = (p[:,i+1] - p[:,i])/del_s
    p_s[:,-1] = p_s[:,-2]


    for i in range(robot.N):
        #print("tensor forces: ", tendon_forces)
        f = robot.rhoAg - Rs[:, :, robot.N-i-1] @ (robot.C * q[:, robot.N-i-1] * np.abs(q[:, robot.N-i-1])) + tendon_forces
        ns = robot.rhoA * Rs[:, :,robot.N-i-1] @ (np.cross(w[:, robot.N-i-1], q[:, robot.N-i-1]) + qt[:, robot.N-i-1]) - f
        if i!=9:
            n[:, robot.N-i-2] = n[:, robot.N-i-1] - ns * robot.L/(robot.N) # compute the internal forces backwards from the tip to the root

    for i in range(robot.N):
        ms = Rs[:, :, robot.N-i-1] @ (np.cross(w[:, robot.N-i-1], robot.rhoJ @ w[:, robot.N-i-1]) + \
                                      robot.rhoJ @ wt[:, robot.N-i-1]) - np.cross(p_s[:, robot.N-i-1], n[:, robot.N-i-1])
        if i!=9:
            m[:, robot.N-i-2] = m[:, robot.N-i-1] - ms * robot.L/(robot.N)  # u_star is all 0 since the quaternions are all the same

    return n, m

def estimate_state(data, tensions, robot):
    """
    Input:
    data [T, 7, n]: the x, y, z positions and 4D quaternions measured at equal spacing, for a total of T steps
    N: integer, rod discretization

    Output:
    estimated_state [T, 25, N], the full 25 states at all N points along the rod
    """
    arc_lengths = np.linspace(0, robot.L, robot.N)  # actual arc lengths


    T, _, n = data.shape
    estimated_state = np.zeros((T, 25, robot.N))
    estimated_state[:,21,:] = 1

    estimated_state[:, :3, :] = data[:, :3, :]  # position, p
    estimated_state[:, :2, 0] = 0  # the base x y are always 0


    estimated_state[:, 3:7, :] = data[:, 3:7, :]  # quaternion, h
    # Estimating q, velocities using numerical differentiation
    velocities = np.gradient(estimated_state[:, :3, :], robot.del_t, axis=0, edge_order=1)
    estimated_state[:, 13:16, :] = velocities  # velocity
    # Estimating w, angular velocities from quaternion derivatives
    angular_velocities = compute_angular_velocities(estimated_state[:, 3:7, :], robot.del_t)
    estimated_state[:, 16:19, :] = angular_velocities  # angular velocity
    # Estimating qt and wt, linear and angular acceleration using numerical differentiation
    qt = np.gradient(velocities, robot.del_t, axis=0, edge_order=2)  # no smoothing
    wt = np.gradient(angular_velocities, robot.del_t, axis=0, edge_order=2) # no smoothing

    # Loop through each time step
    for t in range(T):
        # Extract positions and quaternions
        positions = estimated_state[t, :3, :]
        quaternions = estimated_state[t, 3:7, :]

        # Estimating v and u from p and h
        v, u = compute_v_u(positions, quaternions, arc_lengths)
        if t==0:
            v_prev = v
            u_prev = u
        estimated_state[t, 19:22, 0] = v[:, 0] # linear rate of change, v, at the root
        robot.vstar = estimated_state[0, 19:22, 0]  # setting the undeformed state to be the initial state

        # Estimating n and m
        # Quaternion to Rotation
        rotation_matrices = np.zeros([3, 3, robot.N])  # Shape: (N, 3, 3)
        for i in range(robot.N):
            h = quaternions[:, i]
            h1, h2, h3, h4 = h
            rotation_matrices[:, :, i] = np.eye(3) + 2 / np.dot(h, h) * \
                            np.array([[-h3**2-h4**2, h2*h3-h4*h1, h2*h4+h3*h1],
                            [h2*h3+h4*h1, -h2**2-h4**2, h3*h4-h2*h1],
                            [h2*h4-h3*h1, h3*h4+h2*h1, -h2**2-h3**2]])

        internal_forces, internal_moments = compute_internal_forces_and_moments(positions,
                                                                                arc_lengths,
                                                                                rotation_matrices,
                                                                                velocities[t],
                                                                                angular_velocities[t],
                                                                                qt[t],
                                                                                wt[t],
                                                                                tensions[t],
                                                                                robot)

        # we know the following values at the tip. Hence do not update
        estimated_state[t, 7:10, :-1] = internal_forces[:, :-1]  # internal forces, n
        estimated_state[t, 10:13, :-1] = internal_moments[:, :-1]  # internal moments, m

        for i in range(robot.N):
            # reestimating v using the computed n. v[:,:,0] is already accurate
            vh = robot.c1 * v[:, i] + robot.c2 * v_prev[:, i]
            uh = robot.c1 * u[:, i] + robot.c2 * u_prev[:, i]
            v[:, i] = robot.Kse_plus_c0_Bse_inv @ (rotation_matrices[:, :, i].T @ estimated_state[t, 7:10, i] + robot.Kse_vstar - robot.Bse @ vh)
            u[:, i] = robot.Kbt_plus_c0_Bbt_inv @ (rotation_matrices[:, :, i].T @ estimated_state[t, 10:13, i] - robot.Bbt @ uh)

        estimated_state[t, 19:22, :] = v[:, :] # linear rate of change, v
        estimated_state[t, 22:, :] = u[:, :] # linear rate of change, v
        estimated_state[t, 4:7, 0] = 0 # setting the quaternion h1 h2 h3 h4 to 0 for the root, always pointing upwards

        v_prev = v
        u_prev = u

    return estimated_state

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from cosserat_ode import CosseratRod

    parser = argparse.ArgumentParser(description='Estimate dstate.')
    parser.add_argument('data_name', default='rand_0_60s')
    args = parser.parse_args()
    dataname = args.data_name

    plot_len = 299
    trim_len = 50
    plot_grid_idx = 1

    ref_robot = CosseratRod()
    measured_loc = [0, 3.23, 5.13, 7.07, 9]  # measurement location ratios have big impact on interpolation results
    real_data_partial_state = np.load(f'datas/{dataname}.bag.npy', allow_pickle=True).item()
    print("data keys: ", real_data_partial_state.keys())
    real_data_ori = np.array(real_data_partial_state['orientation']).swapaxes(0, 1).swapaxes(1,2)
    # real_data_pos = np.array(real_data_partial_state['positions']).swapaxes(0, 1).swapaxes(1,2)
    # real_data_traj = np.concatenate([real_data_pos, real_data_ori], 1)
    # print(real_data_ori.shape, real_data_pos.shape, real_data_traj.shape)

    real_data_controls = real_data_partial_state['controls']
    #print("real data traj shape: ", real_data_traj.shape)
    print("real data controls shape: ", real_data_controls.shape)

    partial_grid = np.stack([real_data_partial_state['interpolated'][:, :, 0],
                             real_data_partial_state['interpolated'][:, :, 3],
                             real_data_partial_state['interpolated'][:, :, 5],
                             real_data_partial_state['interpolated'][:, :, 7],
                             real_data_partial_state['interpolated'][:, :, 9]], axis=2)
    real_data_traj_full_grid = fit_curve(partial_grid, measured_loc, 10)
    real_data_traj_full_state = estimate_state(real_data_traj_full_grid, real_data_controls, ref_robot)

    #real_data_traj_full_state = estimate_state(real_data_partial_state['interpolated'][:300], real_data_controls[:300], ref_robot)
    np.save(f'datas/{dataname}_estimated.npy',
            {"traj":real_data_traj_full_state, "controls": real_data_controls})

    # sim_data1 = np.load('data/sim_sin_1_0_amp_300.npy', allow_pickle=True).item()


    # # plot the two trajectories side by side
    # fig = plt.figure()
    # ax_titles = ['x', 'y', 'z', 'h0', 'h1', 'h2', 'h3', 'n1', 'n2', 'n3', 'm1', 'm2', 'm3',
    #              'q1', 'q2', 'q3', 'w1', 'w2', 'w3', 'v1', 'v2', 'v3', 'u1', 'u2', 'u3']

    # for i in range(25):
    #     ax = fig.add_subplot(5, 5, i+1)
    #     ax.plot(real_data_traj_full_state[:plot_len, i, plot_grid_idx], 'r', label='read estimate')
    #     ax.plot(sim_data1['traj'][:, :25][:plot_len, i, plot_grid_idx], 'b', label='sim')
    #     ax.set_title(ax_titles[i])
    #     ax.legend()



    # # the second grid [pos=1] of x, y, z are not good (estimated too large), the rest are ok. z not good overall for lower grids
    # # overall h is good but phase is off, h0's beginning has jumps, h1-h3's roots are hardcoded to 0. h3 is crap
    # # v1 v2 are very good, v3 is good (v3 is derived from n3)
    # # n1 n2 are very good, n3 is good (simulated n3 is a lot noisier)
    # # m's values rise too slowly from the tip to root, but shape is good
    # state_idx = 13
    # fig1 = plt.figure()
    # ax1_titles = []
    # for i in range(10):
    #     ax1_titles.append("grid"+str(i))

    # print("Another: ", real_data_traj_full_state.shape, sim_data1['traj'].shape)

    # for i in range(10):
    #     ax1 = fig1.add_subplot(10,1,i+1)
    #     ax1.plot(real_data_traj_full_state[:plot_len, state_idx, i], 'r', label='real estimate')
    #     ax1.plot(sim_data1['traj'][:, :25][:plot_len, state_idx, i], 'b', label='sim')
    #     ax1.set_title(ax_titles[state_idx] + ' ' + ax1_titles[i])
    #     ax1.legend()

    # plt.show()
