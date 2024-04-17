import numpy as np
from .distance import color_within_0_255
from .lloyd import lloyd
from typing import Union, Tuple

class KRegions:
    """K-regions Algorithm."""
    def __init__(self,
                 k: int=4,
                 max_iter: int=500, 
                 W: np.ndarray=None,
                 gamma: float=0.5,) -> None:
        
        self.k = k
        self.max_iter = max_iter
        self.W = W
        self.gamma = gamma

    def __m_relative_abundance__(self,) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fit(self,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


class KRegionsClassification(KRegions):
    """K-regions algorithm for classification task."""
    def __init__(self,
                 traindata_cls_counts: dict, 
                 position: dict, 
                 d: int,
                 k: int=4,
                 max_iter: int=500, 
                 W: np.ndarray=None,
                 gamma: float=0.5,
                 ) -> None:
        
        super().__init__(k=k, max_iter=max_iter, W=W, gamma=gamma)

        self.traindata_cls_counts = traindata_cls_counts
        self.position = position
        self.d = d

    def __m_relative_abundance__(self,) -> Tuple[np.ndarray, np.ndarray]:
        """
        By default, clients with classification task are scattered in two cities.
        
        Args:
            d: number of features.
        """
        
        assert len(self.traindata_cls_counts) == len(self.position)
        
        n = len(self.traindata_cls_counts)
        C = np.zeros((n, self.d)) # mrtaix of M-Relative Abundance
        V = np.zeros((n, 2)) # matrix of coordinate
        
        for i in range(n):
            for key in self.traindata_cls_counts[i]:
                C[i][key] = self.traindata_cls_counts[i][key]
            V[i][0] = self.position[i][0]
            V[i][1] = self.position[i][1]
        
        # print(C[0])
        
        A = [] # City 1
        B = [] # City 2
        
        for i in range(n):
            if self.position[i][2] == 1:
                A.append(C[i])
            elif self.position[i][2] == 2:
                B.append(C[i])
                
        A, B = np.array(A), np.array(B)
        
        A_mean = np.mean(A, axis=0)
        B_mean = np.mean(B, axis=0)
        
        n_min = np.min(np.array([A_mean, B_mean]), axis=0)
        n_max = np.max(np.array([A_mean, B_mean]), axis=0)
        
        # compute M-Relative Abundance among tow cities
        for i in range(n):
            for j in range(self.d):
                C[i][j] = color_within_0_255(C[i][j], j, n_max, n_min)
                
        assert C.shape[0] == V.shape[0]
        assert C.shape[1] == self.d
        assert V.shape[1] == 2
        
        return V, C


    def fit(self,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        V, C = self.__m_relative_abundance__()
        return lloyd(V, C, k=self.k, max_iter=self.max_iter, W=self.W, gamma=self.gamma)
