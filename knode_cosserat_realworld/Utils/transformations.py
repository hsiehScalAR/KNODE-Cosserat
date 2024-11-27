import torch

def quaternion_to_euler(quaternions):
    """
    Convert a tensor of quaternions to Euler angles (roll, pitch, yaw) with adjusted shape.
    
    Args:
    - quaternions (Tensor): a tensor of shape [4, a] containing quaternions
    
    Returns:
    - Tensor: a tensor of shape [3, a] containing Euler angles
    """
    # Ensure input is a float tensor
    quaternions = quaternions.float()
    
    # Normalize the quaternions
    norms = quaternions.norm(p=2, dim=0, keepdim=True)
    normalized_quaternions = quaternions / norms
    
    # Extract components
    w, x, y, z = normalized_quaternions[0], normalized_quaternions[1], normalized_quaternions[2], normalized_quaternions[3]
    
    # Compute Euler angles
    roll = torch.atan2(2*(w*y + x*z), 1 - 2*(y**2 + z**2))
    pitch = torch.asin(torch.clamp(2*(w*z - x*y), -1.0, 1.0))  # Clamp to avoid NaNs if out of bounds
    yaw = torch.atan2(2*(w*x + y*z), 1 - 2*(x**2 + z**2))
    
    # Combine the angles into a new tensor
    euler_angles = torch.stack([roll, pitch, yaw], dim=0)
    
    return euler_angles



if __name__ == "__main__":
    # Example usage:
    # quaternions is a tensor of shape [4, a] where each row is a component of the quaternion (w, x, y, z)
    # quaternions = torch.tensor([[w1, w2, ...], [x1, x2, ...], [y1, y2, ...], [z1, z2, ...]])
    # euler_angles = quaternion_to_euler_v2(quaternions)
    # print(euler_angles)
    pass    


