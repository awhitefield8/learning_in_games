import numpy as np


def snap(point,grid):
    """
    Args:
        point:
        grid: 
    Returns:
        index of nearest point in grid
    """
    array = np.array(grid)  # Convert your list to a NumPy array
    # Find the index of the nearest point in s_array to the target_float
    return np.abs(array - point).argmin()




def inverse_mw(new_weight,old_weight,nu=0.1):
    """
    Args:
        new_weight: new weight
        old_weight: initial weight
        cost: cost
        nu: learning rate
    Returns:
        cost required to generate new weight
    """
    return (1 - (new_weight/old_weight))/nu

