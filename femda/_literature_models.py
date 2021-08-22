import numpy as np
import pandas as pd
import random, os, time, csv, warnings
from ._algo_utils import fit_t, label_outliers

import math
from scipy import stats, special, optimize, spatial
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model._base import LinearClassifierMixin

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
r = robjects.r
psych = importr('psych')
rrcov = importr('rrcov')
SpatialNP = importr('SpatialNP')
LaplacesDemon = importr('LaplacesDemon')


class LDA(BaseEstimator, LinearClassifierMixin):
    # Attributes:
    # priors: 1xK
    # coefficients: KxM
    # intercepts: #1xK
    # parameters
    # means_ : MxK
    # covariance_: KxMxM
    def __init__(self, method='distributional', pool_covs=True, fudge=1):
        self.method = method

        self.pool_covs = pool_covs
        self.fudge = fudge
    
    def _bose_k(self):
        return np.array([0.5])
    
    def _mahalanobis(self, X, ki=None): #NxM -> KxN
        ret = []
        r = range(self._K) if ki is None else [ki]
        for k in r:
            m = X - self.means_[:,k]
            kth_maha = np.array(list(map(lambda d: d @ np.linalg.inv(self.covariance_)[k,:,:] @ d[:,None], m))).T
            #kth_maha = np.diag(m @ np.linalg.inv(self.covariances)[k,:,:] @ m.T)]
            ret += [kth_maha]
        return np.vstack(ret) if ki is None else ret[0]
    
    def _general_discriminants(self, X): #KxN
        return -0.5*np.log(np.linalg.det(self.covariance_))[:,None] * self.fudge - self._bose_k()[:,None] * self._mahalanobis(X)
    
    def _kth_likelihood(self, k): # non-log likelihood
        return stats.multivariate_normal(mean=self.means_[:,k], cov=self.covariance_[k,:,:])
    
    def _log_likelihoods(self, X):
        r = [self._kth_likelihood(k).pdf(X) for k in range(self._K)]
        return np.log(np.array(r))
       
    def _estimate_parameters(self, X): #NxM -> [1xM, MxM]
        return [X.mean(axis=0), np.cov(X.T)]

    def _dk_from_method(self, X):
        if self.method=='distributional':
            return self._log_likelihoods(X)
        elif self.method=='generalised':
            return self._general_discriminants(X)

    def fit(self, X, y):
        X, y = self._validate_data(X, y, ensure_min_samples=2, estimator=self,
                                   dtype=[np.float64, np.float32])
        st=time.time()

        self._ks = np.unique(y) #1xK
        self._K = len(self._ks)
        self._M = X.shape[1]
        self.X_classes = [X[np.where(y == k), :][0,:,:] for k in self._ks] #Kxn_kxM
        n = np.array([c.shape[0] for c in self.X_classes])
        
        try:
            self._parameters_ = [self._estimate_parameters(c) for c in self.X_classes]
        except np.linalg.LinAlgError:
            return None
        self.means_ = np.array([param[0] for param in self._parameters_]).T
        self.covariance_ = np.array([param[1] for param in self._parameters_])
        self.covariance_ = np.repeat(np.sum(n[:,None,None] * self.covariance_, axis=0)[None,:],self._K,axis=0) / n.sum() \
                if self.pool_covs else self.covariance_ 
                    
        self.priors_ = n / n.sum()
        #print(self.priors, n)

        assert(n.sum() == X.shape[0])
        assert(self._M == self.covariance_.shape[2])
        assert (np.allclose(self.priors_.sum(), 1))
        #print("Fitting time", time.time()-st)
        return self
                    
    def predict(self, X, percent_outliers=0):
        if self._parameters_ is None:
            return None
        try:
            dk = self._dk_from_method(X)
        except np.linalg.LinAlgError:
            return None
        #print("Before priors", dk)
        #self.priors = np.array([1/6, 1/6, 1/6, 1/6, 0.0001, 1/6, 1/6])
        dk = dk + np.log(self.priors_[:, None])
        #print("After priors", dk)
        #check priors fitted in all algos
        #print(self.priors)
        y = self._ks[np.nanargmax(dk, axis=0)]
        return label_outliers(X, y, self.means_, self.covariance_, thres=percent_outliers)
       
    def predict_proba(self, X):
        dk = self._dk_from_method(X)
        if self.method != 'distributional':
            dk = np.exp(dk)
        return (dk/dk.sum(axis=0)).T
    

## Custom QDA
class QDA(LDA):
    def __init__(self, method='distributional'):
        super().__init__(method)
        self.pool_covs = False
        
## t-LDA (mostly redundant: pooling covs still means discriminant depends on x^2 because of differing dofs (assuming same dofs = MMD classifier)
class t_LDA(LDA):
    def __init__(self, method='distributional'):
        super().__init__(method)
        self.dofs = None #1xK
    
    def _kth_likelihood(self, k):
        return stats.multivariate_t(loc=self.means_[:,k], shape=self.covariance_[k,:,:], df=self.dofs[k])
    
    def estimate_parameters(self, X):
        return fit_t(X)
    
    def _bose_k(self):
        return (0.5*(1 + self._M/self.dofs))
    
    def _discriminants(self, X): #NxM -> KxN
        return None
        
    def _general_discriminants(self, X):
        v = self.dofs
        return super()._general_discriminants(X) + (special.gammaln((v+self._M)/2) - special.gammaln(v/2) - 0.5*self._M*np.log(v))[:,None]
            
    def fit(self, X,y):
        super().fit(X,y)
        if self._parameters_ is None:
            return None
        self.dofs = np.array([param[2] for param in self._parameters_]).squeeze()

## t-QDA
class t_QDA(t_LDA):
    def __init__(self, method='distributional'):
        super().__init__(method)
        self.pool_covs = False

## GQDA selon Bose et al.
class GQDA(QDA):
    def __init__(self):
        super().__init__(method='generalised')
        self.c = None
    
    def fit(self, X,y,c=None):
        super().fit(X,y) #Kx[n_k, M]
        if self._parameters_ is None:
            return None
        
        if c is not None:
            self.c = c
            return 
            
        uijs = [np.zeros((self.X_classes[k].shape[0], self._K, self._K)) for k in range(self._K)] #Kx[n_kxIxJ]
        sij = np.zeros((self._K,self._K))
        logdets = np.log(np.linalg.det(self.covariance_)) #K,  
        for i in range(self._K):
            for j in range(self._K):
                dij_on_i = self._mahalanobis(self.X_classes[i], ki=j) - self._mahalanobis(self.X_classes[i], ki=i) #Kxn_i
                dij_on_j = self._mahalanobis(self.X_classes[j], ki=j) - self._mahalanobis(self.X_classes[j], ki=i) #Kxn_j
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
        for e,c in enumerate(T):
            
            for i in range(self._K):
                Rijc = []
                for j in range(self._K):
                    if i==j: continue
                    p = uijs[i][:, i,j]
                    to_app = p > -c if sij[i,j]>0 else p < -c 
                    Rijc.append(self.X_classes[i][to_app])
                Rijc = np.vstack(Rijc)
                Ric = np.unique(Rijc, axis=0)
                #print(Ric.shape, Rijc.shape)
                lenRic = Ric.shape[0]
                MCic = self.X_classes[i].shape[0] - lenRic
                #print(MCic, Ric.shape)
                MCc[e] += MCic
                
        #return uijs, MCc, T
        c_star = T[MCc.argmin()]
        self.c = c_star if c_star > 0 else 0.001
        print("optimal c is", c_star)
        
    def _bose_k(self):
        return np.array([0.5/self.c])
        

## RGQDA (becomes classical QDA with robust estimator if c=1) 
class RGQDA(GQDA):
    def __init__(self, estimation='gaussian'):
        super().__init__()
        self.estimation = estimation
        
    def estimate_parameters(self, X): #NxM -> [1xM, MxM]
        if self.estimation == 'gaussian':
            return self.estimate_gaussian_MLE(X)
        elif self.estimation == 't-EM':
            return self.estimate_t_EM(X)
        elif self.estimation == 'winsorised':
            return self.estimate_winsorised(X)
        elif self.estimation == 'MVE':
            return self.estimate_MVE(X)
        elif self.estimation == 'MCD':
            return self.estimate_MCD(X)
        elif self.estimation == 'M-estimator':
            return self.estimate_M_estimator(X)
        elif self.estimation == 'S-estimator':
            return self.estimate_S_estimator(X)
        elif self.estimation == 'SD-estimator':
            return self.estimate_SD_estimator(X)

    def estimate_t_EM(self, X):
        return fit_t(X) #discarding dof parameters

    def estimate_gaussian_MLE(self, X):
        return [X.mean(axis=0), np.cov(X.T)]

    def _get_r_frame(self, X):
        return pandas2ri.py2rpy(pd.DataFrame(X))

    def estimate_winsorised(self, X):
        frame = self._get_r_frame(X)
        winsorised = psych.winsor(frame, trim=0.1)
        return self.estimate_gaussian_MLE(winsorised)

    def estimate_MVE(self, X):
        frame = self._get_r_frame(X)
        MVE = rrcov.CovMve(frame, alpha=0.5)
        return [MVE.slots['center'], MVE.slots['cov']]

    def estimate_MCD(self, X):
        print("estimating...")
        frame = self._get_r_frame(X)
        MCD = rrcov.CovMcd(frame, alpha=0.5)
        return [MCD.slots['center'], MCD.slots['cov']]   
    
    def estimate_S_estimator(self, X):
        frame = self._get_r_frame(X)
        S = rrcov.CovSest(frame)
        return [S.slots['center'], S.slots['cov']]  

    def estimate_SD_estimator(self, X):
        frame = self._get_r_frame(X)
        SD = rrcov.CovSde(frame)
        return [SD.slots['center'], SD.slots['cov']] 
    
    def estimate_M_estimator(self, X):
        frame = self._get_r_frame(X)
        M = SpatialNP.mvhuberM(frame)
        #print(list(M))
        return list(M)   

