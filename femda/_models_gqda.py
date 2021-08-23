import numpy as np
import pandas as pd

from ._algo_utils import fit_t
from ._models_lda import QDA

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
r = robjects.r
psych = importr('psych')
rrcov = importr('rrcov')
SpatialNP = importr('SpatialNP')
LaplacesDemon = importr('LaplacesDemon')

## GQDA selon Bose et al.
class GQDA(QDA):
    def __init__(self):
        super().__init__(method='generalised')
    
    def _bose_k(self):
        return np.array([0.5/self.c_])

    def fit(self, X, y, c_=None):
        super().fit(X,y) #Kx[n_k, M]
        
        if c_ is not None:
            self.c_ = c_
            return self
            
        uijs = [np.zeros((self.X_classes_[k].shape[0], self._K, self._K)) for k in range(self._K)] #Kx[n_kxIxJ]
        sij = np.zeros((self._K,self._K))
        logdets = np.log(np.linalg.det(self.covariance_)) #K,  
        for i in range(self._K):
            for j in range(self._K):
                dij_on_i = self._mahalanobis(self.X_classes_[i], ki=j) - self._mahalanobis(self.X_classes_[i], ki=i) #Kxn_i
                dij_on_j = self._mahalanobis(self.X_classes_[j], ki=j) - self._mahalanobis(self.X_classes_[j], ki=i) #Kxn_j
                sij[i,j] = logdets[j] - logdets[i]
                
                uijs[i][:, i, j] = dij_on_i / sij[i,j]
                uijs[i][:, j, j] = np.inf
                uijs[j][:, i, j] = dij_on_j / sij[i,j]
        
        T = []
        for uij in uijs:
            T.append(uij[(uij > 0) * (uij<1)])
        T = np.sort(np.concatenate(T))
        T = np.concatenate([np.array([0]), T])
        #print(T)
        MCc = np.zeros((len(T)))
        for e,c_ in enumerate(T):
            for i in range(self._K):
                Rijc = []
                for j in range(self._K):
                    if i==j: continue
                    p = uijs[i][:, i,j]
                    to_app = p > -c_ if sij[i,j]>0 else p < -c_ 
                    Rijc.append(self.X_classes_[i][to_app])
                Rijc = np.vstack(Rijc)
                Ric = np.unique(Rijc, axis=0)
                #print(Ric.shape, Rijc.shape)
                lenRic = Ric.shape[0]
                MCic = self.X_classes_[i].shape[0] - lenRic
                #print(MCic, Ric.shape)
                MCc[e] += MCic
                
        #return uijs, MCc, T
        c_star = T[MCc.argmin()]
        self.c_ = c_star if c_star > 0 else 0.001
        print("optimal c is", c_star)
        return self        

## RGQDA (becomes classical QDA with robust estimator if c_=1) 
class RGQDA(GQDA):
    def __init__(self, estimation='gaussian'):
        super().__init__()
        self.estimation = estimation
        
    def _estimate_parameters(self, X): #NxM -> [1xM, MxM]
        if self.estimation == 'gaussian':
            return self._estimate_gaussian_MLE(X)
        elif self.estimation == 't-EM':
            return self._estimate_t_EM(X)
        elif self.estimation == 'winsorised':
            return self._estimate_winsorised(X)
        elif self.estimation == 'MVE':
            return self._estimate_MVE(X)
        elif self.estimation == 'MCD':
            return self._estimate_MCD(X)
        elif self.estimation == 'M-estimator':
            return self._estimate_M_estimator(X)
        elif self.estimation == 'S-estimator':
            return self._estimate_S_estimator(X)
        elif self.estimation == 'SD-estimator':
            return self._estimate_SD_estimator(X)

    def _estimate_t_EM(self, X):
        return fit_t(X) #discarding dof parameters

    def _estimate_gaussian_MLE(self, X):
        return [X.mean(axis=0), np.cov(X.T)]

    def _get_r_frame(self, X):
        return pandas2ri.py2rpy(pd.DataFrame(X))

    def _estimate_winsorised(self, X):
        frame = self._get_r_frame(X)
        winsorised = psych.winsor(frame, trim=0.1)
        return self._estimate_gaussian_MLE(winsorised)

    def _estimate_MVE(self, X):
        frame = self._get_r_frame(X)
        MVE = rrcov.CovMve(frame, alpha=0.5)
        return [MVE.slots['center'], MVE.slots['cov']]

    def _estimate_MCD(self, X):
        print("estimating...")
        frame = self._get_r_frame(X)
        MCD = rrcov.CovMcd(frame, alpha=0.5)
        return [MCD.slots['center'], MCD.slots['cov']]   
    
    def _estimate_S_estimator(self, X):
        frame = self._get_r_frame(X)
        S = rrcov.CovSest(frame)
        return [S.slots['center'], S.slots['cov']]  

    def _estimate_SD_estimator(self, X):
        frame = self._get_r_frame(X)
        SD = rrcov.CovSde(frame)
        return [SD.slots['center'], SD.slots['cov']] 
    
    def _estimate_M_estimator(self, X):
        frame = self._get_r_frame(X)
        M = SpatialNP.mvhuberM(frame)
        #print(list(M))
        return list(M)   

