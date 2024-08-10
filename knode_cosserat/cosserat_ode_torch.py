import numpy as np
import torch
import torch.nn as nn

class CosseratRodTorch:
    def __init__(self, device, n_layers, nn_input_history=False):
        self.device = device
        self.use_nn = True
        self.nn_input_history = nn_input_history
        self.verbose = False
        self.y = None # robot state [p;h;n;m;q;w]
        self.z = None # state related to time [v;u]
        # Parameters - Section 2.1
        self.L = 0.4 # length
        self.N = 10 # number of elements (spatial discretization)
        self.E = 109e9 # Young's modulus
        self.r = 0.0012 # rod radius
        self.rho = 8000. # rod density
        self.vstar = torch.tensor([0., 0., 1.], device=self.device) # value of v when rod is straight, and n=v_t=0
        self.g = torch.tensor([0, 0, -9.81], device=self.device) # gravity
        self.Bse = torch.zeros((3,3), device=self.device) # Damping matrix for shear and extension
        self.Bbt = torch.diag(torch.tensor([3e-2, 3e-2, 3e-2], device=self.device)) # Damping matrix for bending and twisting
        self.C = torch.tensor([1e-4, 1e-4, 1e-4], device=self.device) # Square law drag coefficients matrix
        self.del_t = 0.005 # time step for BDF2
        self.F_tip = torch.zeros(3, device=self.device) # tip force is zero
        self.M_tip = torch.zeros(3, device=self.device) # tip moment is zero

        # tendons
        self.T0 = 5  # baseline tendon tension
        self.n_tendons = 4  # number of tendons
        self.tendon_tensions = None
        theta = torch.tensor(torch.pi) / self.n_tendons  # tendon angle (using pi value since torch doesn't have a constant for it)
        self.tendon_offset = 0.02  # offset of the tendons from the center
        self.tendon_dirs = torch.tensor([
            [torch.cos(theta), torch.sin(theta), 0],
            [torch.cos(theta + torch.pi / 2), torch.sin(theta + torch.pi / 2), 0],
            [torch.cos(theta + torch.pi), torch.sin(theta + torch.pi), 0],
            [torch.cos(theta + 3 * torch.pi / 2), torch.sin(theta + 3 * torch.pi / 2), 0]
        ]).to(self.device)  # tendon directions

        # Boundary Conditions - Section 2.4
        self.p0 = torch.zeros(3, requires_grad=True).to(self.device) # initial position
        self.h0 = torch.tensor([1., 0., 0., 0.], requires_grad=True).to(self.device) # initial quaternion
        self.q0 = torch.zeros(3, requires_grad=True).to(self.device) # initial velocity
        self.w0 = torch.zeros(3, requires_grad=True).to(self.device) # initial angular velocity

        self.compute_intermediate_terms()

        self.residualArgs = {"yh": None,
                             "zh": None,
                             "tendon_forces": None}

        # activ = nn.Tanh()
        activ = nn.Softplus()
        #activ = nn.Tanh()
        # neural networks

        # 11 layers for long

        self.layers = [nn.Linear(53 if self.nn_input_history else 28, n_layers),
                       nn.ELU(),
                       nn.Linear(n_layers, 25)]
        """
        self.layers = [nn.Linear(53, 64),
                       activ,
                       nn.Linear(64, 128),
                       activ,
                       nn.Linear(128, 256),
                       activ,
                       nn.Linear(256, 64),
                       activ,
                       nn.Linear(64, 25)]
        """

        # initialize weights and biases
        for i in range(len(self.layers)):
            if str(self.layers[i])[:6] == 'Linear':
                #torch.nn.init.xavier_uniform_(self.layers[i].weight)
                self.non_negative_normal_init(self.layers[i], mean=0.01, std=0.01)
                #torch.nn.init.xavier_normal_(self.layers[i].weight)
                #torch.nn.init.kaiming_uniform_(self.layers[i].weight)
                #torch.nn.init.constant_(self.layers[i].weight, 0.01)
                #self.layers[i].bias.data.fill_(0.01)
                nn.init.normal_(self.layers[i].bias, mean=0.0, std=0.01)


        # initialize the model
        self.nn_models = nn.ModuleList(self.layers).to(self.device)

    def non_negative_normal_init(self, m, mean, std):
        """
        Initializes the weights of a PyTorch layer with non-negative values from a normal distribution.

        Args:
        - m: The PyTorch layer to initialize.
        - mean: The mean of the normal distribution (non-negative).
        - std: The standard deviation of the normal distribution.
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # Ensure mean is non-negative
            assert mean >= 0, "Mean must be non-negative"

            # Initialize weights
            with torch.no_grad():
                m.weight.data.normal_(mean, std).abs_()  # Apply abs to ensure non-negativity


    def compute_intermediate_terms(self):
        # Dependent Parameter Calculations
        self.A = torch.pi * self.r**2 # cross-sectional area
        self.G = self.E / (2 * (1 + 0.3)) # shear modulus
        self.ds = self.L / (self.N - 1) # spatial discretization

        self.J = torch.diag(torch.tensor([torch.pi*self.r**4/4, torch.pi*self.r**4/4, torch.pi*self.r**4/2])).to(self.device) # Second mass moment of inertia tensor
        self.Kse = torch.diag(torch.tensor([self.G*self.A, self.G*self.A, self.E*self.A])).to(self.device) # Extension and shear stiffness matrix
        self.Kbt = torch.diag(torch.tensor([self.E*self.J[0,0], self.E*self.J[1,1], self.G*self.J[2,2]])).to(self.device) # Bending and twisting stiffness matrix

        # BDF2 Coefficients
        self.c0 = 1.5 / self.del_t
        self.c1 = -2 / self.del_t
        self.c2 = 0.5 / self.del_t

        # Expressions extracted from simulation loop
        self.Kse_plus_c0_Bse_inv = torch.inverse(self.Kse + self.c0 * self.Bse)
        self.Kbt_plus_c0_Bbt_inv = torch.inverse(self.Kbt + self.c0 * self.Bbt)
        self.Kse_vstar = torch.matmul(self.Kse, self.vstar)
        self.rhoA = self.rho * self.A
        self.rhoAg = self.rho * self.A * self.g
        self.rhoJ = self.rho * self.J

    def forward(self, x):
        for _, l in enumerate(self.nn_models):
            x = l(x)
        return x


    def ODE(self, y, yh, zh, tendon_forces):
        """
        yh is y history, 19-dimensional, stacked

        y is 19-dimensional
        y[:3], or p, is the position
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
        R = torch.eye(3).to(self.device) + 2 / torch.dot(h, h) * \
            torch.tensor([[-h3**2-h4**2, h2*h3-h4*h1, h2*h4+h3*h1],
                          [h2*h3+h4*h1, -h2**2-h4**2, h3*h4-h2*h1],
                          [h2*h4-h3*h1, h3*h4+h2*h1, -h2**2-h3**2]], requires_grad=True).float().to(self.device)

        # Solved Constitutive Law - Eq(6)
        v = torch.mv(self.Kse_plus_c0_Bse_inv, (torch.mv(R.T, n) + self.Kse_vstar - torch.mv(self.Bse, vh)))
        u = torch.mv(self.Kbt_plus_c0_Bbt_inv, (torch.mv(R.T, m) - torch.mv(self.Bbt, uh)))
        z = torch.cat([v, u], 0)

        # Time Derivatives - Eq(5)
        yt = self.c0 * y + yh
        zt = self.c0 * z + zh
        vt, ut, qt, wt = zt[0:3], zt[3:6], yt[13:16], yt[16:19]

        # Weight and Square-Law-Drag - Eq(3)
        f = self.rhoAg - torch.mv(R, (self.C * q * torch.abs(q))) + tendon_forces

        # Rod State Derivatives - Eq(7)
        ps = torch.mv(R, v)
        ns = self.rhoA * torch.mv(R, (torch.cross(w, q) + qt)) - f
        ms = torch.mv(R, (torch.cross(w, self.rhoJ @ w) + self.rhoJ @ wt)) - torch.cross(ps, n)
        qs = vt - torch.cross(u, q) + torch.cross(w, v)
        ws = ut - torch.cross(u, w)


        # Quaternion Derivative - Eq(9)
        hs_mat = torch.tensor([[0, -u[0], -u[1], -u[2]],
                               [u[0], 0, u[2], -u[1]],
                               [u[1], -u[2], 0, u[0]],
                               [u[2], u[1], -u[0], 0]], requires_grad=True).to(self.device)
        hs = 0.5 * torch.mv(hs_mat, h)
        ys = torch.cat([ps, hs, ns, ms, qs, ws], 0)

        if self.use_nn:
            # hybrid neural network
            if self.nn_input_history:
                nn_input = torch.cat([y, yh, z, zh, tendon_forces], 0)
            else:
                nn_input = torch.cat([y, z, tendon_forces], 0)
            nn_output = self.forward(nn_input)
            ys_nn = nn_output[:19] #* self.traj_range[:19] + self.traj_min[:19]
            """
            nn_input = torch.cat([(y-self.traj_min[:19])/self.traj_range[:19],
                                  (yh-self.traj_min[:19])/self.traj_range[:19],
                                  (z-self.traj_min[19:])/self.traj_range[19:],
                                  (zh-self.traj_min[19:])/self.traj_range[19:],
                                  (self.tendon_tensions-self.controls_min)/self.controls_range], 0)

            nn_output = self.forward(nn_input)
            ys_nn = nn_output[:19] #* self.traj_range[:19] + self.traj_min[:19]
            z_nn = nn_output[19:] #* self.traj_range[19:] + self.traj_min[19:]
            """
            ys += ys_nn
            z += nn_output[19:]

        return ys, z


    def ODE_parallel(self, ys, yhs, zhs, tendon_forcess):
        """
        yhs is y history, Q x 19-dimensional, stacked

        ys is Q x 19-dimensional
        ys[:, :3], or p, is the position
        ys[:, 3:7], or h, is the quaternion
        ys[:, 7:10], or n,  are internal forces
        ys[:, 10:13], or m, are internal moments
        ys[:, 13:16], or q, is the velocity in the local frame
        ys[:, 16:19], or w, is the angular velocity in the local frame

        zhs is Q x 6-dimensional, are the history terms
        zh[:, 0:3], or vh, is the history terms for v
        zh[:, 3:6], or uh, is the history terms for u

        tendon_forcess is a Q x T array of tendon forces
        """
        Q = ys.shape[0]

        # ys, zhs, and zhs are vectors. To make stuff easier with Torch's bmm,
        # I am going to keep these as stacks of N x 1 vectors.
        ys = ys.unsqueeze(2)
        ys_in = ys
        yhs = yhs.unsqueeze(2)
        zhs = zhs.unsqueeze(2)
        tendon_forcess = tendon_forcess.unsqueeze(2)

        # Split the y tensor into the corresponding components
        h = ys[:, 3:7]
        n = ys[:, 7:10]
        m = ys[:, 10:13]
        q = ys[:, 13:16]
        w = ys[:, 16:19]

        vh = zhs[:, 0:3]
        uh = zhs[:, 3:6]

        # Quaternion to Rotation - Eq(10)
        h_norm = torch.sum(h**2, dim=1, keepdim=True)
        hflat = h.squeeze(2)
        h1, h2, h3, h4 = hflat[:, 0], hflat[:, 1], hflat[:, 2], hflat[:, 3]

        Qtimes = lambda x: x.unsqueeze(0).repeat(Q, 1, 1) # Duplicate a mxn matrix Q times
        Qtimesv = lambda x: x.unsqueeze(0).repeat(Q, 1).unsqueeze(2) # Duplicates a vector Q times
        crossv = lambda x, y: torch.cross(x.squeeze(2), y.squeeze(2), dim=1).unsqueeze(2)

        R = Qtimes(torch.eye(3).to(ys.device)) + 2 / h_norm * \
            torch.stack([
                torch.stack([-h3**2-h4**2, h2*h3-h4*h1, h2*h4+h3*h1], dim=1),
                torch.stack([h2*h3+h4*h1, -h2**2-h4**2, h3*h4-h2*h1], dim=1),
                torch.stack([h2*h4-h3*h1, h3*h4+h2*h1, -h2**2-h3**2], dim=1)
            ], dim=1)

        RT = R.transpose(1, 2)

        # Solved Constitutive Law - Eq(6)
        v = torch.bmm(Qtimes(self.Kse_plus_c0_Bse_inv), torch.bmm(RT, n) + self.Kse_vstar.unsqueeze(1) - torch.bmm(Qtimes(self.Bse), vh))
        u = torch.bmm(Qtimes(self.Kbt_plus_c0_Bbt_inv), torch.bmm(RT, m) - torch.bmm(Qtimes(self.Bbt), uh))

        z = torch.cat([v, u], dim=1)

        if self.verbose:
            print("computed z in ODE_parallel: ", z)

        # Time Derivatives - Eq(5)
        yt = self.c0 * ys + yhs
        zt = self.c0 * z + zhs
        vt, ut, qt, wt = zt[:, 0:3], zt[:, 3:6], yt[:, 13:16], yt[:, 16:19]

        # Weight and Square-Law-Drag - Eq(3)

        f = Qtimesv(self.rhoAg) - torch.bmm(R, Qtimesv(self.C) * q * torch.abs(q)) + tendon_forcess

        # Rod State Derivatives - Eq(7)
        ps = torch.bmm(R, v)
        ns = self.rhoA * torch.bmm(R, crossv(w, q) + qt) - f

        ms = torch.bmm(R, crossv(w, torch.bmm(Qtimes(self.rhoJ), w)) + torch.bmm(Qtimes(self.rhoJ), wt)) - crossv(ps, n)
        qs = vt - crossv(u, q) + crossv(w, v)
        ws = ut - crossv(u, w)

        # Quaternion Derivative - Eq(9)
        zeros = torch.zeros(Q, 1).to(ys.device)
        hs_mat = torch.stack([torch.stack([zeros, -u[:, 0], -u[:, 1], -u[:, 2]], dim=1),
                              torch.stack([u[:, 0], zeros, u[:, 2], -u[:, 1]], dim=1),
                              torch.stack([u[:, 1], -u[:, 2], zeros, u[:, 0]], dim=1),
                              torch.stack([u[:, 2], u[:, 1], -u[:, 0], zeros], dim=1)], dim=1).squeeze(3)
        hs = 0.5 * torch.bmm(hs_mat, h)
        ys = torch.cat([ps, hs, ns, ms, qs, ws], dim=1)

        if self.use_nn:
            # hybrid neural network
            if self.nn_input_history:
                nn_input = torch.cat([ys_in, yhs, z, zhs, tendon_forcess], dim=1)
            else:
                nn_input = torch.cat([ys_in, z, tendon_forcess], dim=1)
            nn_output = self.forward(nn_input.squeeze(2)).unsqueeze(2)
            ys_nn = nn_output[:, :19]
            # print('wx before nn', ys[:, 16])
            # print('nn for wx', ys_nn[:, 16])

            ys += ys_nn
            z += nn_output[:, 19:]

        return ys.squeeze(2), z.squeeze(2)


    def getResidualEuler(self, G):
        y, z, yh, zh= self.y, self.z, self.residualArgs["yh"], self.residualArgs["zh"]

        # Reaction force and moment are guessed
        n0 = G[0:3]
        m0 = G[3:6]

        y_new = torch.cat([self.p0, self.h0, n0, m0, self.q0, self.w0]).to(self.device)
        y = torch.cat([y_new.unsqueeze(1), y[:, 1:]], dim=1).float()
        tendon_forces = torch.matmul(self.tendon_tensions, self.tendon_dirs).to(self.device)  # 3x1 vector of forces in x,y,z
        # concatenate y and z to represent the full rod
        full_rod = [torch.cat([y[:, 0], z[:, 0]], dim=0)]
        # Euler Integration
        for j in range(self.N-1):
            yj = y[:,j]
            # Get dy (change in y) from ODE and update y using Euler's method
            if self.verbose:
                print("ode input: ", yj, yh[:,j], zh[:,j], tendon_forces)
            dy, zj_new = self.ODE(yj, yh[:,j], zh[:,j], tendon_forces)
            #################################################################
            y_next = yj + self.ds * dy
            #################################################################
            y = torch.cat((y[:,:j+1], y_next.unsqueeze(1), y[:,j+2:]), dim=1)
            z = torch.cat((z[:,:j], zj_new.unsqueeze(1), z[:,j+1:]), dim=1)
            # print the sizes of y and z
            if self.verbose:
                print("y size in residual: ", y.shape)
                print("z size in residual: ", z.shape)
            full_rod.append(torch.cat([y_next, zj_new], dim=0))

        full_rod = torch.stack(full_rod, dim=1)
        # Cantilever boundary conditions
        nL = y[7:10,-1]
        mL = y[10:13,-1]
        self.y = y

        if self.verbose: print("output from Residual ", nL, mL)
        NF = self.F_tip - nL # net force, should be zero
        NM = self.M_tip - mL # net moment, should be zero

        # Sum of squares of residuals
        total_residual = torch.sum(NF**2 + NM**2)
        return total_residual, full_rod


    def getNextSegmentEuler(self,  G):
        yh, zh= self.residualArgs["yh"], self.residualArgs["zh"]

        # Reaction force and moment are guessed for the next state
        y = G[:19, :]
        z = G[19:, :]

        tendon_forces = torch.matmul(self.tendon_tensions, self.tendon_dirs).to(self.device)  # 3x1 vector of forces in x,y,z
        # concatenate y and z to represent the full rod
        full_rod = [torch.cat([y[:, 0], z[:, 0]], dim=0)]

        # Euler Integration
        for j in range(self.N-1):
            yj = y[:,j]
            if self.verbose:
                print("in ode: ", torch.isnan(yj).any())
                print("ode input: ", yj, yh[:,j], zh[:,j], tendon_forces)
            # Get dy (change in y) from ODE and update y using Euler's method
            dy, zj_new = self.ODE(yj, yh[:,j], zh[:,j], tendon_forces)
            y_next = yj + self.ds * dy

            # notice that y and z are not updated here
            # print the sizes of y and z
            if self.verbose:
                print("y size in residual: ", y.shape)
                print("z size in residual: ", z.shape)
            full_rod.append(torch.cat([y_next, zj_new], dim=0))

        full_rod = torch.stack(full_rod, dim=1)
        return full_rod

    def parallelGetNextSegmentEuler(self,  Gs, segment_idxs, args):
        yhs, zhs = args["yh"], args["zh"]
        full_rods = []

        G_T = Gs.transpose(1, 2)
        yh_T = yhs.transpose(1, 2)
        zh_T = zhs.transpose(1, 2)

        tendon_forces = torch.matmul(args["tendon_tensions"], self.tendon_dirs)

        # Flatten segment + keypoint indices together
        all_ys = torch.flatten(G_T[:, segment_idxs - 1, :19], end_dim=1)
        all_yh = torch.flatten(yh_T[:, segment_idxs - 1], end_dim=1)
        all_zh = torch.flatten(zh_T[:, segment_idxs - 1], end_dim=1)
        all_tendon_forces = []
        for stp_idx in range(Gs.shape[0]):
            all_tendon_forces.append(tendon_forces[stp_idx].repeat((segment_idxs.shape[0], 1)))
        all_tendon_forces = torch.flatten(torch.stack(all_tendon_forces, dim=0), end_dim=1)

        all_dys, all_zjs_new = self.ODE_parallel(all_ys, all_yh, all_zh, all_tendon_forces)
        all_ys_next = all_ys + self.ds * all_dys

        # Unflatten everything
        all_ys_next = all_ys_next.reshape((Gs.shape[0], -1, *all_ys_next.shape[1:]))
        all_zjs_new = all_zjs_new.reshape((Gs.shape[0], -1, *all_zjs_new.shape[1:]))

        # return torch.cat([all_ys_next, all_zjs_new], dim=2).transpose(1, 2)

        for stp_idx in range(Gs.shape[0]):
            ys_next = all_ys_next[stp_idx]
            zjs_new = all_zjs_new[stp_idx]
            catted = torch.cat([ys_next, zjs_new], dim=1)

            full_rod_at_keypoints = torch.stack([c for c in catted], dim=1)
            full_rods.append(full_rod_at_keypoints)

        return torch.stack(full_rods)

if __name__ == "__main__":
    device = "cuda"
    #save_nn_path = "saved_models/zero_init_nn.pth"
    save_nn_path = None

    #robot = CosseratRodTorch()
    robot = torch.load("saved_models/quick_test.pth")["robot"]
    robot.use_nn = True
    check_idx = 13
    np_data = np.load("data/random_traj_2000_steps_10_grid_E209.npy", allow_pickle=True).item()
    #np_data = np.load("data/quick_test.npy", allow_pickle=True).item()

    #np_data = np.load("data/random_traj_300_steps_15grid.npy", allow_pickle=True).item()
    traj = torch.tensor(np_data["traj"]).float().to(device)
    controls = torch.tensor(np_data["controls"]).float().to(device)
    #data = torch.tensor(np.load("data/high_damping_500steps.npy")).float().to(device)
    #data = torch.tensor(np.load("data/smallE.npy")).float().to(device)
    print("training traj has shape: ", traj.shape)
    print("training control has shape: ", controls.shape)

    if save_nn_path is not None:
        torch.save({'robot': robot}, save_nn_path)
    # y is general ODE state vector
    # z is general vector with relevant time derivatives, but elements not in y
    y = traj[check_idx, 0:19, :]
    z = traj[check_idx, 19:, :]

    robot.y = y  # Updating the robot state to y
    robot.z = z
    if check_idx == 0:
        y_prev = robot.y.clone()
        z_prev = robot.z.clone()
    else:
        y_prev = traj[check_idx-1, 0:19, :]
        z_prev = traj[check_idx-1, 19:, :]

    # Main Simulation Loop - Section 2.4
    # G = torch.zeros(6, requires_grad=True)  # Shooting method initial guess
    G = traj[check_idx+1, 7:13, 0]

    robot.tendon_tensions = controls[check_idx]  # Control inputs
    # Set history terms - Eq(5)
    yh = robot.c1 * robot.y + robot.c2 * y_prev
    zh = robot.c1 * robot.z + robot.c2 * z_prev

    y_prev = robot.y.clone()
    z_prev = robot.z.clone()

    robot.residualArgs["yh"] = yh
    robot.residualArgs["zh"] = zh

    residual, full_rod = robot.getResidualEuler(G)
    print("residual: ", residual)
    print("full rod shape: ", full_rod.shape)
