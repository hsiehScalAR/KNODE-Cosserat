import numpy as np
import torch
from scipy.optimize import fsolve, minimize
from cosserat_ode import CosseratRod

def setup_robot(robot, mod=None, original=False):
    """Set up robot based on expeirmental parameters"""
    if original:
        raise Exception("--original parameter no longer supported")
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
