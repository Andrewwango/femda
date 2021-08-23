import numpy as np
import math

from ._fem import FEM
from ._algo_utils import fit_t_dof, fit_t
from ._models_lda import LDA
from ._models_t_lda import t_LDA

class _FEM_classification(FEM):
    def _e_step_indicator(self, X):
        return np.ones((X.shape[0], self.K))
    def _e_step(self, X):
        return super()._e_step(X)

    def fit(self, X):
        pass
    def override_params(self, mu, Sigma):
        self.mu_, self.Sigma_ = mu, Sigma
    def update_tau(self, X):
        p = X.shape[1]
        tau_new = np.ones((X.shape[0], self.K))
        for k in range(self.K):
            
            diff = X - self.mu_[k]
            tau_new[:, k] = (np.dot(diff, np.linalg.inv(self.Sigma_[k])) * diff).sum(1) / p
            tau_new[:, k] = np.where(tau_new[:, k] < 10**(-12) , 10**(-12),
            np.where(tau_new[:, k] > 10**(12), 10**(12), tau_new[:, k]))
        self.tau_ = tau_new.copy()
    

class _LDA_FEM_base():
    def __init__(self):
        self._mean = None
        self._scatter = None
        self._cov = None
        
    def _scatter_constant(self, X):
        return X.shape[1]/np.trace(self._scatter)

    def _FEM_estimate(self, X): #X is {X: I(Zi=k|Xi=xi)=1}
        # initialise K=2 (for 1 class, but to maintain structure), tau,alpha randomly and means and sigma with Gaussian MLE
        _K = 2
        FEM_estimator = _FEM_classification(_K, rand_initialization=True)
        FEM_estimator._initialize(X)
        FEM_estimator.override_params(np.repeat(X.mean(axis=0)[None,:], _K, axis=0), np.repeat(np.cov(X.T)[None,:], _K, axis=0))
        
        #run E-step to get indicators
        cond_prob = FEM_estimator._e_step_indicator(X)
        #run M-step
        FEM_estimation = FEM_estimator._m_step(X, cond_prob)  
        self._mean = FEM_estimation[1][0,:]; self._scatter = FEM_estimation[2][0,:,:]
        s = self._scatter_constant(X)

        self._cov = self._scatter * s
        #print("scatter->cov:", s)
        return [self._mean, self._cov]

class LDA_FEM(LDA, _LDA_FEM_base):
    def _estimate_parameters(self, X):
        return self._FEM_estimate(X)

class QDA_FEM(LDA_FEM, _LDA_FEM_base):
    def __init__(self, method='distributional'):
        super().__init__(method=method, pool_covs=False)
        
class t_LDA_FEM(t_LDA, _LDA_FEM_base):
    def _estimate_parameters(self, X):
        params = self._FEM_estimate(X)
        return params + [fit_t_dof(X, *params, dof_0=3)]
    
class t_QDA_FEM(t_LDA_FEM, _LDA_FEM_base):
    def __init__(self, method='distributional'):
        super().__init__(method=method, pool_covs=False)
        
class FEMDA(QDA_FEM):
    def _log_likelihoods2(self, X): #->KxN
        FEM = _FEM_classification(self._K, rand_initialization=True)
        FEM._initialize(X)
        FEM.override_params(self.means_.T, self.covariance_)
        FEM.update_tau(X)
        cond_prob = FEM._e_step(X)
        return cond_prob.T
    
    def _simple_mahalanobis(self, X, m, S):
        diff = X - m
        return (np.dot(diff, np.linalg.inv(S)) * diff).sum(1)
    
    def _log_likelihoods(self, X):
        N,p = X.shape
        log_maha = np.zeros((self._K, N))
        for k in range(self._K):
            log_maha[k, :] = np.log(self._simple_mahalanobis(X, self.means_[:,k], self.covariance_[k,:,:]))
        _,logdets = np.linalg.slogdet(self.covariance_)
        pik = -0.5 * (p * log_maha + logdets[:,None])
        return pik

class FEMDA_N(FEMDA):

    def _normalise_centered(self, X, y, mean_estimator=fit_t):
        def _normalise(a):
            return (a.T/np.linalg.norm(a, axis=1)).T
        X_copy = X.copy()
        for k in np.unique(y):
            mean = mean_estimator(X[y==k])[0]
            X_copy[y==k] = _normalise(X[y==k]-mean)+mean
        return X_copy

    def fit(self, X, y):
        X_n = self._normalise_centered(X, y, mean_estimator=lambda x:_LDA_FEM_base()._FEM_estimate(x))
        return super().fit(X_n,y)

# don't know what this is for
class FEM_predictor(_FEM_classification):
    def __init__(self, K):
        super().__init__(K, rand_initialization=True)
    
    def predict(self, X_new, DA_means, DA_covs):
        assert(self.K == DA_covs.shape[0])
        self._initialize(X_new)
        self.override_params(DA_means.T, DA_covs)

        return super().predict(X_new, thres=0)

