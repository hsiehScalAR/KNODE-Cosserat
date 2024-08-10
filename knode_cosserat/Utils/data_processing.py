import numpy as np

def normalize_data(data):
    """
    Normalize the data along the last axis so that each value lies between 0 and 1.
    Works for both 2D and 3D data.

    Parameters:
    data (numpy array): A numpy array of shape (x, z) or (x, y, z).

    Returns:
    numpy array: Normalized data.
    numpy array: Minimum values of the original data for the last axis.
    numpy array: Maximum values of the original data for the last axis.
    """

    # Determine the axis to perform normalization
    if data.ndim == 2:
        axis = (0) # average across all time
    elif data.ndim == 3:
        axis = (0,2) # average across all time and space

    # Find the minimum and maximum values along the last axis
    min_vals = np.min(data, axis=axis, keepdims=True)
    max_vals = np.max(data, axis=axis, keepdims=True)

    # Normalize the data
    # Using np.clip to avoid division by zero by ensuring range_vals is never zero
    range_vals = np.clip(max_vals - min_vals, 1e-10, np.inf)
    normalized_data = (data - min_vals) / range_vals
    return normalized_data, min_vals.squeeze(), range_vals.squeeze()

def denormalize_data(normalized_data, min_vals, max_vals):
    """
    De-normalize the normalized data back to its original range.
    Works for both 2D and 3D data.

    Parameters:
    normalized_data (numpy array): Normalized data of shape (x, z) or (x, y, z).
    min_vals (numpy array): Minimum values of the original data for the last axis.
    max_vals (numpy array): Maximum values of the original data for the last axis.

    Returns:
    numpy array: De-normalized data.
    """

    # De-normalize the data
    denormalized_data = normalized_data * (max_vals - min_vals) + min_vals

    return denormalized_data
