import torch


def eulers_method(robot, y, yh, zh, tendon_tensions, device='cuda'):
    n_tendons = 4  # number of tendons
    theta = torch.tensor(torch.pi) / n_tendons  # tendon angle (using pi value since torch doesn't have a constant for it)
    tendon_dirs = torch.tensor([
        [torch.cos(theta), torch.sin(theta), 0],
        [torch.cos(theta + torch.pi / 2), torch.sin(theta + torch.pi / 2), 0],
        [torch.cos(theta + torch.pi), torch.sin(theta + torch.pi), 0],
        [torch.cos(theta + 3 * torch.pi / 2), torch.sin(theta + 3 * torch.pi / 2), 0]
    ]).to(device)  # tendon directions
    tendon_forces = torch.matmul(tendon_tensions, tendon_dirs).to(self.device)  # 3x1 vector of forces in x,y,z

    spatial_pred = torch.zeros((25, robot.N-1)).to(device)
    for j in range(robot.N-1):
        yj = y[:,j]
        dy, zj_new = robot.ODE(yj, yh[:,j], zh[:,j], tendon_forces)
        y_next = yj + robot.ds * dy
        spatial_pred[:, j] = (torch.cat([y_next, zj_new], dim=0))