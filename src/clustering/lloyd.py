import numpy as np
from typing import Union, Tuple
from .kmeans_pp import cdist, kmeans_pp

def lloyd(V: np.ndarray, 
          C: np.ndarray, 
          k: int = 4, 
          init_way: str = 'kmeans_pp', 
          max_iter: int = 500, 
          W: np.ndarray=None,
          gamma: int = 0.5
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lloyd's Algorithm.

    Args:
        V: coordinate matrix.
        C: matrix of M-relative abundance.
        k: number of clusters.
        init_way: the way to initialize the cluster center and defualt value is `kmeans_pp`.
        max_iter: number of maximal iterations.
        W: weights matrix of feature.
        gamma: hyperparameter.

    return: coordinate matrix of center, matrix of M-relative abundance of center, cluster identifier, clustering structure. 
    """

    if init_way == 'kmeans_pp':
        inits_V, inits_C, _ = kmeans_pp(V, C, k=k, W=W, gamma=gamma)
    else:
        raise RuntimeError(f'{init_way} is not an initial way.')

    n, d = C.shape
    
    Sr_list = []
    for _ in range(max_iter):
        
        pd = cdist(inits_V, inits_C, V, C, W=W, gamma=gamma)
        assert pd.shape[0] == inits_V.shape[0] and pd.shape[1] == V.shape[0]
        pd = pd**2

        Sr_list = []
        for r in range(k):
            th = pd[r, :]
            remaining_dist = pd[np.arange(k) != r]
            assert np.allclose(remaining_dist.shape, [k - 1, n])
            indicator = (remaining_dist - th) < 0
            indicator = np.sum(indicator.astype(int), axis=0)
            assert len(indicator) == n
            # places where indicator is 0 is our set
            Sr = [i for i in range(len(indicator)) if indicator[i] == 0]
            assert len(Sr) >= 0
            Sr_list.append(Sr)
        # We don't mind lloyd_init being dense. Its only k x d.
        inits_V = np.array([np.mean(V[Sr], axis=0) for Sr in Sr_list])
        inits_C = np.array([np.mean(C[Sr], axis=0) for Sr in Sr_list])
    
        assert np.allclose(inits_V.shape, [k, V.shape[1]])
        assert np.allclose(inits_C.shape, [k, C.shape[1]])
        
    cluster_identifier = np.zeros((n,), dtype=int)
    
    for cid in range(k):
        for idx in Sr_list[cid]:
            cluster_identifier[idx] = cid
            
    return inits_V, inits_C, cluster_identifier, np.array(Sr_list, dtype=object)
    