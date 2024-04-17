import numpy as np
from typing import Union, Tuple
from .distance import region_wise_distance


def cdist(A_V: np.ndarray, A_C: np.ndarray,
          S_V: np.ndarray, S_C: np.ndarray,
          W: np.ndarray=None,
          gamma: float = 0.5) -> np.ndarray:
    n, m = A_C.shape[0], S_C.shape[0]
    res = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            res[i][j] = region_wise_distance(A_V[i], S_V[j], A_C[i], S_C[j], W=W, gamma=gamma)
    
    return res

def distance_to_set(A_V: np.ndarray, A_C: np.ndarray,
                    S_V: np.ndarray, S_C: np.ndarray, 
                    W: np.ndarray=None,
                    gamma: float = 0.5) -> np.ndarray:
    '''
    S is a list of points. Distance to set is the minimum distance of `x` to
    points in `S`. In this case, this is computed for each row of `A`. 
    
    Returns a single array of length len(A) containing corresponding distances.
    '''

    n, d = A_C.shape
    assert A_V.ndim == 2 and S_V.ndim == 2
    assert A_V.shape[1] == 2 and S_V.shape[1] == 2
    assert A_C.ndim == 2 and S_C.ndim == 2
    assert A_C.shape[1] == d and S_C.shape[1] == d
    
    
    pd = cdist(A_V, A_C, S_V, S_C, W=W, gamma=gamma)
    
    assert np.allclose(pd.shape, [A_C.shape[0], S_C.shape[0]])
    
    dx = np.min(pd, axis=1)
    assert len(dx) == A_C.shape[0]
    assert dx.ndim == 1
    return dx


def kmeans_pp(A_V: np.ndarray, A_C: np.ndarray, k: int, 
              weighted: bool=True, verbose: bool=False, 
              W: np.ndarray=None,
              gamma: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KMeans++ Algorithm.
    Returns `k` initial centers based on the k-means++ initialization scheme.
    With weighted set to True, we have the standard algorithm. When weighted is
    set to False, instead of picking points based on the D^2 distribution, we
    pick the farthest point from the set (careful deterministic version --
    affected by outlier points). Note that this is not deterministic.

    Args:
        A_V: nx2 coordinate matrix (dense)
        A_C: nxd data matrix (dense). 
        k: is the number of clusters.

    Returns a (k x d) dense matrix.
    """
    
    n, d = A_C.shape
    if n <= k:
        return A_V, A_C, np.array([i for i in range(n)])
    
    index = np.random.choice(n)
    
    inits_V = [A_V[index]]
    inits_C = [A_C[index]]

    indices = [index]
    t = [x for x in range(A_C.shape[0])]
    distance_matrix = distance_to_set(A_V, A_C, np.array(inits_V), np.array(inits_C), W=W, gamma=gamma)
    distance_matrix = np.expand_dims(distance_matrix, axis=1)
    
    while len(inits_V) < k and len(inits_C) < k:
        if verbose:
            print('\rCenter: %3d/%4d' % (len(inits_V) + 1, k), end='')
        # Instead of using distance to set we can compute this incrementally.
        dx = np.min(distance_matrix, axis=1)
        assert dx.ndim == 1
        assert len(dx) == n
        dx = dx**2/np.sum(dx**2)
        if weighted:
            choice = np.random.choice(t, 1, p=dx)[0]
        else:
            choice = np.argmax(dx)
        if choice in indices:
            continue
        temp_V = A_V[choice]
        temp_C = A_C[choice]
        inits_V.append(temp_V)
        inits_C.append(temp_C)
        indices.append(choice)
        
        last_center_V = np.expand_dims(temp_V, axis=0)
        last_center_C = np.expand_dims(temp_C, axis=0)
        
        assert last_center_V.ndim == 2 and last_center_C.ndim == 2
        assert last_center_V.shape[0] == 1 and last_center_C.shape[0] == 1
        assert last_center_V.shape[1] == 2 and last_center_C.shape[1] == d
        dx = distance_to_set(A_V, A_C, last_center_V, last_center_C, gamma=gamma)
        assert dx.ndim == 1
        assert len(dx) == n
        dx = np.expand_dims(dx, axis=1)
        a = [distance_matrix, dx]
        distance_matrix = np.concatenate(a, axis=1)
        
    if verbose:
        print()
    return np.array(inits_V), np.array(inits_C), np.array(indices)