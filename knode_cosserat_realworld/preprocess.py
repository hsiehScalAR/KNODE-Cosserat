from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

def position(data, link):
    return np.array([data[f'{link}.pose.position.x'], data[f'{link}.pose.position.y'], data[f'{link}.pose.position.z']])

def adj_pos(positions):
    adjusted_positions = positions.copy() - positions[0] # Subtract base position from all links
    adjusted_positions[1:, 2] += 0.0635 # Base markers are this high up from the bottom of the robot
    # Because the bottom of the rod is anchored at the bottom of the robot, the bottom link does not get shifted up
    return adjusted_positions

def preprocessed(merged, ts=None):
    LINKS = ['base', 'link0', 'link1', 'link2', 'link3']
    positions = np.stack([position(merged, link).T for link in LINKS])
    orientations = np.stack([fix_orientations(merged, link) for link in LINKS])
    if ts is None:
        interpolated = np.stack([interpolate_posquat(adj_pos(positions[:, t]), orientations[:, t], 10)
                                 for t in range(positions.shape[1])])
        return positions, orientations, interpolated
    else:
        s_positions = np.stack([
            np.stack([
                np.interp(ts, merged.Time, q) for q in p.T
            ]).T for p in positions
        ])

        s_orientations = np.stack([
            Slerp(merged.Time, Rotation.concatenate(r))(ts)
            for r in orientations
        ])
        interpolated = np.stack([interpolate_posquat(adj_pos(s_positions[:, t]), s_orientations[:, t], 10)
                                 for t in range(s_positions.shape[1])])

        # Adjust the positions. Because adj_positions needs 2d arrays, process timestep by timestep
        f_positions = np.array([adj_pos(s_positions[:, t]) for t in range(s_positions.shape[1])])


        return f_positions, s_orientations, interpolated

def interpolate_posquat_ryan(adjusted_positions, quaternions, N):
    s = [0, 3/9, 5/9, 7/9, 1] # Positions of links along the rod, as a fraction of rod length
    ts = np.linspace(0, 1, N) # Positions to interpolate at

    # 2D interpolation for position
    position_spline = CubicSpline(s, adjusted_positions)

    # 1D Interpolation for quaternion
    quaternion_spline = Slerp(s, Rotation.concatenate(quaternions))
    quaternion_spline = np.vstack([r.as_quat(canonical=True, scalar_first=True) for r in quaternion_spline(ts)])

    return np.concatenate([position_spline(ts).T, quaternion_spline.T])


def guess_fix(rotation):
    """Fit a matrix to axis permutation, or 45 degree rotation + axes permutation."""
    matrix = rotation.as_matrix()

    # Try axes permutation
    rfix = matrix.round(0)
    if np.array_equal(rfix @ rfix.T, np.eye(3)):
        return Rotation.from_matrix(rfix)

    # Try 45 degree rotation + axes permutation
    # This has issues!
    # rotation_45 = Rotation.from_rotvec([0, 0, np.pi/4]).as_matrix()
    # candidate = matrix @ rotation_45
    # rfix = candidate.round(0)
    # if np.array_equal(rfix @ rfix.T, np.eye(3)):
    #     corrected = Rotation.from_matrix(rfix @ rotation_45.T)
    #     error = (rotation.inv() * corrected).magnitude()
    #     print(error * 180 / np.pi)
    #     if error < 10 * np.pi / 180:
    #         return corrected

    # Give up and return the original rotation
    return rotation

def fix_orientations(data, link):
    quat = np.array([
        data[f'{link}.pose.orientation.x'],
        data[f'{link}.pose.orientation.y'],
        data[f'{link}.pose.orientation.z'],
        data[f'{link}.pose.orientation.w']
    ]).T
    rotations = [Rotation.from_quat(q) for q in quat]

    # Fix first rotation.
    rfix = (rotations[0].inv()).as_matrix().round(0)
    if np.array_equal(rfix @ rfix.T, np.eye(3)):
        rotations[0] = rotations[0] * Rotation.from_matrix(rfix)

    # Fix all other rotations so they align with the previous rotation
    rprev = rotations[0]
    for i in range(1, len(rotations)):
        rnext = rotations[i]
        if not rprev.approx_equal(rnext, atol=30, degrees=True):
            rnext = rnext * guess_fix(rnext.inv() * rprev)
            rotations[i] = rnext
        rprev = rnext

    return rotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp

def compute_tangent_vectors(positions, new_grid, order=5):
    """
    Compute tangent vectors from positions using polynomial fitting.

    Parameters:
    - positions: array of shape [T, 3, n], positions at measurement points
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
        interpolated_quats[t, :, :] = interpolated_rots.as_quat(scalar_first=True, canonical=True).T

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

def interpolate_posquat_tom(adjusted_positions, quaternions, N):
    measurement_loc = [0, 3/9, 5/9, 7/9, 1] # Positions of links along the rod, as a fraction of rod length
    q = Rotation.concatenate(quaternions).as_quat(canonical=True)
    measured_poses = np.concatenate([adjusted_positions.T, q.T])[np.newaxis, :]
    return fit_curve(measured_poses, measurement_loc, N)[0]

interpolate_posquat = interpolate_posquat_ryan
