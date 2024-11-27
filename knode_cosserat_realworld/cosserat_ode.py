import numpy as np
import torch

class CosseratRod:
    def __init__(self,
                 nn_path=None,
                 use_fsolve=False,
                 nn_input_history=False):

        self.verbose = False
        self.use_fsolve = use_fsolve
        self.nn_path = nn_path
        self.nn_input_history = nn_input_history
        # Parameters - Section 2.1
        self.L = 0.4 # length
        self.N = 10 # number of elements (spatial discretization)
        self.E = 109e9 #207e9 # Young's modulus
        self.r = 0.0012 # rod radius
        self.rho = 8000 # rod density
        self.vstar = np.array([0, 0, 1]) # value of v when rod is straight, and n=v_t=0
        self.g = np.array([0, 0, -9.81])  # gravity
        self.Bse = np.zeros((3,3)) # Damping matrix for shear and extension
        # self.Bbt = np.zeros((3,3)) # Damping matrix for bending and twisting
        # self.C = np.zeros(3) # Square law drag coefficients matrix
        self.Bbt = np.diag([3e-2, 3e-2, 3e-2])  # Material damping coefficients - bending and torsion
        self.C = np.array([1e-4, 1e-4, 1e-4])    # Square-law-drag damping coefficients
        self.del_t = 0.005 # time step for BDF2
        self.F_tip = np.zeros(3) # tip force is zero
        self.M_tip = np.zeros(3) # tip moment is zero

        # tendons
        self.T0 = 5 # baseline tendon tension
        self.n_tendons = 4 # number of tendons
        self.tendon_tensions = None
        theta = np.pi/self.n_tendons # tendon angle
        self.tendon_offset = 0.02 # offset of the tendons from the center
        self.tendon_dirs = np.array([
            [np.cos(theta), np.sin(theta), 0],
            [np.cos(theta + np.pi/2), np.sin(theta + np.pi/2), 0],
            [np.cos(theta + np.pi), np.sin(theta + np.pi), 0],
            [np.cos(theta + 3*np.pi/2), np.sin(theta + 3*np.pi/2), 0]]) # tendon directions

        # Boundary Conditions - Section 2.4
        self.p0 = np.zeros(3) # initial position
        self.h0 = np.array([1, 0, 0, 0]) # initial quaternion
        self.q0 = np.zeros(3) # initial velocity
        self.w0 = np.zeros(3) # initial angular velocity

        self.compute_intermediate_terms()

        if self.nn_path is not None:
            self.nn_model, self.param_ls = self.get_nn_from_file()
        if self.verbose:
            print("nn_model: ", self.nn_model)
            print("param_ls: ", self.param_ls)


    def compute_intermediate_terms(self):
        # Dependent Parameter Calculations
        self.A = np.pi * self.r**2 # cross-sectional area
        self.G = self.E / (2 * (1 + 0.3)) # shear modulus
        self.ds = self.L / (self.N - 1) # spatial discretization
        self.J = np.diag([np.pi*self.r**4/4, np.pi*self.r**4/4, np.pi*self.r**4/2]) # Second mass moment of inertia tensor
        self.Kse = np.diag([self.G*self.A, self.G*self.A, self.E*self.A]) # Extension and shear stiffness matrix
        self.Kbt = np.diag([self.E*self.J[0,0], self.E*self.J[1,1], self.G*self.J[2,2]]) # Bending and twisting stiffness matrix

        # BDF2 Coefficients
        self.c0 = 1.5 / self.del_t
        self.c1 = -2 / self.del_t
        self.c2 = 0.5 / self.del_t

        # Expressions extracted from simulation loop
        self.Kse_plus_c0_Bse_inv = np.linalg.inv(self.Kse + self.c0 * self.Bse)
        self.Kbt_plus_c0_Bbt_inv = np.linalg.inv(self.Kbt + self.c0 * self.Bbt)
        self.Kse_vstar = self.Kse @ self.vstar
        self.rhoA = self.rho * self.A
        self.rhoAg = self.rho * self.A * self.g
        self.rhoJ = self.rho * self.J


    def get_nn_from_file(self):
        nn_model = torch.load(self.nn_path, map_location=torch.device('mps'))['robot'].nn_models # load the neural network
        param_ls = []

        for _, layer in nn_model.state_dict().items():
            param_ls.append(layer.detach().cpu().numpy())

        return nn_model, param_ls

    def get_nn_output(self, input, model, param_ls):
        ode_nn = input # input to the neural network
        softplus = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
        relu = lambda x: np.maximum(0, x)
        elu = lambda x: np.where(x > 0, x, np.exp(x) - 1)
        n_layers = len(model)
        param_cnt = 0
        for i in range(n_layers):
            # hidden layers
            if str(model[i]) == 'Tanh()':
                ode_nn = np.tanh(ode_nn)
            elif str(model[i]) == 'Softplus(beta=1.0, threshold=20.0)':
                ode_nn = softplus(ode_nn)
            elif str(model[i]) == 'ReLU()':
                ode_nn = relu(ode_nn)
            elif str(model[i]) == 'ELU(alpha=1.0)':
                ode_nn = elu(ode_nn)
            elif str(model[i]).startswith('Dropout('):
                ode_nn = ode_nn
            else:
                ode_nn = param_ls[param_cnt]@ ode_nn + param_ls[param_cnt + 1]
                param_cnt += 2
        return ode_nn

    def ODE(self, y, yh, zh, tendon_forces):
        """
        yh is y history, 19-dimensional, stacked

        y is 19-dimensional
        y[3:7], or h, is the quaternion
        y[7:10], or n,  are internal forces
        y[10:13], or m, are internal moments
        y[13:16], or q, is the velocity in the local frame
        y[16:19], or w, is the angular velocity in the local frame

        zh is 6-dimensional, are the history terms
        zh[0:3], or vh, is the history terms for v
        zh[3:6], or uh, is the history terms for u
        """
        h, n, m = y[3:7], y[7:10], y[10:13]
        q, w, vh, uh = y[13:16], y[16:19], zh[0:3], zh[3:6]

        # Quaternion to Rotation - Eq(10)
        h1, h2, h3, h4 = h
        R = np.eye(3) + 2 / np.dot(h, h) * \
            np.array([[-h3**2-h4**2, h2*h3-h4*h1, h2*h4+h3*h1],
                      [h2*h3+h4*h1, -h2**2-h4**2, h3*h4-h2*h1],
                      [h2*h4-h3*h1, h3*h4+h2*h1, -h2**2-h3**2]])

        # Solved Constitutive Law - Eq(6)
        v = self.Kse_plus_c0_Bse_inv @ (R.T @ n + self.Kse_vstar - self.Bse @ vh)
        u = self.Kbt_plus_c0_Bbt_inv @ (R.T @ m - self.Bbt @ uh)
        z = np.hstack([v, u])

        if self.verbose: print("computed z in ODE: ", z)
        # Time Derivatives - Eq(5)
        yt = self.c0 * y + yh
        zt = self.c0 * z + zh
        vt, ut, qt, wt = zt[0:3], zt[3:6], yt[13:16], yt[16:19]

        # Weight and Square-Law-Drag - Eq(3)
        f = self.rhoAg - R @ (self.C * q * np.abs(q)) + tendon_forces

        # Rod State Derivatives - Eq(7)
        ps = R @ v
        ns = self.rhoA * R @ (np.cross(w, q) + qt) - f
        ms = R @ (np.cross(w, self.rhoJ @ w) + self.rhoJ @ wt) - np.cross(ps, n)
        qs = vt - np.cross(u, q) + np.cross(w, v)
        ws = ut - np.cross(u, w)

        # Quaternion Derivative - Eq(9)
        hs_mat = np.array([[0, -u[0], -u[1], -u[2]],
                           [u[0], 0, u[2], -u[1]],
                           [u[1], -u[2], 0, u[0]],
                           [u[2], u[1], -u[0], 0]])
        hs = 0.5 * hs_mat @ h
        ys = np.hstack([ps, hs, ns, ms, qs, ws])

        # ys is 19-dimensional, z is 6-dimensional
        if self.nn_path is not None:

            # form neural network input
            if self.nn_input_history:
                nn_input = np.hstack([y, yh, z, zh, tendon_forces])
            else:
                nn_input = np.hstack([y, z, tendon_forces])
            # get neural network output
            nn_output = self.get_nn_output(nn_input, self.nn_model, self.param_ls)
            ys_nn = nn_output[:19]
            ys += ys_nn


            z_nn = nn_output[19:]
            # print(nn_output)
            z += z_nn

        return ys, z

    def getResidualEuler(self, G, y, z, yh, yh_int, zh, zh_int):
        # the mid points yh_int and zh_int aren't used in Euler's method
        # Reaction force and moment are guessed
        n0 = G[0:3]
        m0 = G[3:6]
        # only update the first grid for y with boundary conditions
        y[:,0] = np.hstack([self.p0, self.h0, n0, m0, self.q0, self.w0])
        tendon_forces = np.dot(self.tendon_tensions, self.tendon_dirs)

        # Euler's Method Integration
        for j in range(self.N-1):
            yj = y[:,j]
            dyds, z[:,j] = self.ODE(yj, yh[:,j], zh[:,j], tendon_forces)
            y[:,j+1] = yj + self.ds * dyds

        # Cantilever boundary conditions, after spatial integration
        nL = y[7:10,-1]
        mL = y[10:13,-1]
        NF = self.F_tip - nL  # net force
        NM = self.M_tip - mL  # net moment
        total_residual = np.sum(NF**2 + NM**2)

        if self.use_fsolve:
            return np.hstack([NF, NM])
        else:
            return total_residual

    def getResidualRK4(self, G, y, z, yh, yh_int, zh, zh_int):
        # Reaction force and moment are guessed
        n0 = G[0:3]
        m0 = G[3:6]
        y[:,0] = np.hstack([self.p0, self.h0, n0, m0, self.q0, self.w0])
        tendon_forces = np.dot(self.tendon_tensions, self.tendon_dirs)

        # Fourth-Order Runge-Kutta Integration
        for j in range(self.N-1):
            yj = y[:,j]
            yhj_int = yh_int[:,j]
            if self.verbose:
                # print the input to the ODE below
                print("yj: ", yj)
                print("yh: ", yh[:, j])
                print("zh: ", zh[:, j])
                print("tendon_forces: ", tendon_forces)
            k1, z[:,j] = self.ODE(yj, yh[:,j],zh[:,j], tendon_forces)
            k2, _ = self.ODE(yj + k1 * self.ds / 2, yhj_int, zh_int[:,j], tendon_forces)
            k3, _ = self.ODE(yj + k2 * self.ds / 2, yhj_int, zh_int[:,j], tendon_forces)
            k4, _ = self.ODE(yj + k3 * self.ds, yh[:,j+1], zh[:,j+1], tendon_forces)
            if self.verbose:
                # print all ks
                print("k1: ", k1)
                print("k2: ", k2)
                print("k3: ", k3)
                print("k4: ", k4)
            y[:,j+1] = yj + self.ds * (k1 + 2 * (k2 + k3) + k4) / 6

        # Cantilever boundary conditions
        nL = y[7:10,-1]
        mL = y[10:13,-1]
        if self.verbose: print("Output from getResidual: ", nL, mL)
        NF = self.F_tip - nL  # net force
        NM = self.M_tip - mL  # net moment
        total_residual = np.sum(NF**2 + NM**2)

        if self.use_fsolve:
            return np.hstack([NF, NM])
        else:
            return total_residual


if __name__ == "__main__":
    # initialize the robot
    nn_path = None
    nn_path = "saved_models/quick_test3.pth"
    robot = CosseratRod(nn_path=nn_path)
    robot.use_fsolve = True
    check_idx = 13
    np_data = np.load("data/random_traj_2000_steps_10_grid_E209.npy", allow_pickle=True).item()
    #np_data = np.load("data/quick_test.npy", allow_pickle=True).item()
    traj = np_data["traj"]
    controls = np_data["controls"]
    #data = np.load("data/high_damping_500steps.npy")
    # print data shape
    print("trajectory shape: ", traj.shape)

    # Initialize to straight configuration
    # y is general ODE state vector
    # y:= [p; h; n; m; q; w]

    y = traj[check_idx, 0:19, :]
    z = traj[check_idx, 19:, :]
    if check_idx == 0:
        y_prev = y
        z_prev = z
    else:
        y_prev = traj[check_idx-1, 0:19, :]
        z_prev = traj[check_idx-1, 19:, :]

    # Main Simulation Loop - Section 2.4
    # G = np.zeros(6)  # Shooting method initial guess
    G = traj[check_idx+1, 7:13, 0]

    # controls
    robot.tendon_tensions = controls[check_idx]

    # Set history terms - Eq(5)
    yh = robot.c1 * y + robot.c2 * y_prev
    zh = robot.c1 * z + robot.c2 * z_prev
    y_prev = y.copy()
    z_prev = z.copy()

    # Midpoints are linearly interpolated for RK4
    yh_int = 0.5 * (yh[:, :-1] + yh[:, 1:])
    zh_int = 0.5 * (zh[:, :-1] + zh[:, 1:])
    residual = robot.getResidualEuler(G, y, z, yh, yh_int, zh, zh_int)
    if robot.use_fsolve:
        NF = residual[0:3]
        MF = residual[3:6]
        print("result: ", np.sum(NF**2 + MF**2))
    else:
        print("result: ", residual)
