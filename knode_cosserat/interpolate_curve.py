import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.spatial.transform import Rotation as R, Slerp

def compute_tangent_vectors(positions, new_grid, order=5):
    """
    Compute tangent vectors from positions using polynomial fitting.

    Parameters:
    - posiations: array of shape [T, 3, n], positions at measurement points
    - new_grid: array of shape [N], new grid for interpolation
    - order: the order of the polynomial to fit

    Returns:
    - tangents: array of shape [T, 3, N], tangent vectors at interpolated points
    """
    T, _, n = positions.shape
    tangents = np.zeros((T, 3, new_grid.size))

    for t in range(T):
        for i in range(3):
            # Fit a polynomial to the data points
            p = np.polyfit(np.linspace(0, 1, n), positions[t, i, :], order)
            # Compute the derivative of the polynomial
            dp = np.polyder(p)
            # Evaluate the derivative at the new grid points to get the tangents
            tangents[t, i, :] = np.polyval(dp, new_grid)

    return tangents

def interpolate_quaternions(measured_quats, measurement_loc, new_grid):
    """
    Interpolate quaternions using SLERP.

    Parameters:
    - measured_quats: array of shape [T, 4, n], initial quaternions at measurement points
    - measurement_loc: array of shape [n], the location of the measurement points
    - new_grid: array of shape [N], new grid for interpolation

    Returns:
    - interpolated_quats: array of shape [T, 4, N], interpolated quaternions
    """
    T, _, n = measured_quats.shape
    interpolated_quats = np.zeros((T, 4, new_grid.size))

    for t in range(T):
        q_initial = R.from_quat(measured_quats[t, :, :].T)
        slerp = Slerp(measurement_loc, q_initial)
        interpolated_rots = slerp(new_grid)
        interpolated_quats[t, :, :] = interpolated_rots.as_quat().T

        # Ensure quaternions are normalized
        for i in range(new_grid.size):
            interpolated_quats[t, :, i] /= np.linalg.norm(interpolated_quats[t, :, i])

    return interpolated_quats

def fit_curve(measured_poses, measurement_loc, N):
    """
    Fit a curve to the measured poses and interpolate smoothly along a grid of N points.

    Parameters:
    - measured_poses [T, 7, n]: T is the total number of time steps, and n is the total number of measurement points.
      7 states in axis=1 corresponds to the positions and orientations (in quaternion)
    - measurement_loc [n,]: the location of the measurement points
    - N: the total number of points to interpolate along the way

    Returns:
    - interpolated_poses [T, 7, N]: the poses on N grid interpolated from the measurements.
      The first and last points along N correspond to the first and last points in the measured poses.
    """
    T, _, n = measured_poses.shape
    interpolated_poses = np.zeros((T, 7, N))

    # Define the new grid for interpolation
    new_grid = np.linspace(measurement_loc[0], measurement_loc[-1], N)

    for t in range(T):
        # Interpolate positions (first 3 states)
        for i in range(3):
            cs = CubicSpline(measurement_loc, measured_poses[t, i, :], bc_type='natural')
            #cs = make_interp_spline(measurement_loc, measured_poses[t, i, :], k=5)
            interpolated_poses[t, i, :] = cs(new_grid)

    # Compute tangent vectors from the interpolated positions
    interpolated_positions = interpolated_poses[:, :3, :]
    tangents = compute_tangent_vectors(interpolated_positions, new_grid)

    # Extract initial quaternions from measured poses
    initial_quaternions = measured_poses[:, 3:, :]

    # Interpolate quaternions using SLERP
    interpolated_quats = interpolate_quaternions(initial_quaternions, measurement_loc, new_grid)

    # Combine positions and quaternions
    interpolated_poses[:, 3:, :] = interpolated_quats

    return interpolated_poses