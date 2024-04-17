import numpy as np


def color_within_0_255(x, idx, n_max, n_min):
    """Color mapping function."""
    c = (x - n_min[idx]) / (n_max[idx] - n_min[idx] + 1e-5)
    c = 0 if c < 1e-5 else c
    c = 1 if c > 1 else c
    return int(c * 255)

def spatial_distance(V_i: np.ndarray, V_j: np.ndarray) -> np.ndarray:
    """Spatial Distance."""

    return np.linalg.norm(V_i - V_j)

def high_dimensional_color_distance(C_i: np.ndarray, C_j: np.ndarray, W: np.ndarray=None) -> np.ndarray:
    """High-dimensional Color Distance."""

    if W is None:
        return np.linalg.norm(C_i - C_j)
    
    assert np.allclose(W.shape, [C_i.shape[0], C_i.shape[0]])

    diff = np.expand_dims(C_i - C_j, axis=0)
    diff = np.abs(diff)

    # print(diff @ W @ diff.T)
    return np.sqrt(diff @ W @ diff.T)[0][0]

def region_wise_distance(V_i: np.ndarray, V_j: np.ndarray,
                         C_i: np.ndarray, C_j: np.ndarray,
                         W: np.ndarray=None,
                         gamma: float=0.5) -> np.ndarray:
    """Region-Wise Distance."""

    return spatial_distance(V_i, V_j) + gamma * high_dimensional_color_distance(C_i, C_j, W=W)