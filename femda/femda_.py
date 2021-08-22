import numpy as np
import math
from ._fem import FEM
from ._algo_utils import fit_t_dof, fit_t
from ._literature_models import LDA, t_LDA

def normalise(a):
    return (a.T/np.linalg.norm(a, axis=1)).T

def normalise_means(means_dict):
    return dict([(k,v/np.linalg.norm(v)) for (k,v) in means_dict.items()])

def normalise_centered(X, y, mean_estimator=fit_t):
    X_copy = X.copy()
    for k in np.unique(y):
        mean = mean_estimator(X[y==k])[0]
        X_copy[y==k] = normalise(X[y==k]-mean)+mean
    return X_copy

class FEM_classification(FEM):
    def fit(self, X):
        pass
    def override_params(self, mu, Sigma):
        self.mu_, self.Sigma_ = mu, Sigma
    def _e_step_indicator(self, X):
        return np.ones((X.shape[0], self.K))
    def _e_step(self, X):
        return super()._e_step(X)
    def update_tau(self, X):
        p = X.shape[1]
        tau_new = np.ones((X.shape[0], self.K))
        for k in range(self.K):
            
            diff = X - self.mu_[k]
            tau_new[:, k] = (np.dot(diff, np.linalg.inv(self.Sigma_[k])) * diff).sum(1) / p
            tau_new[:, k] = np.where(tau_new[:, k] < 10**(-12) , 10**(-12),
            np.where(tau_new[:, k] > 10**(12), 10**(12), tau_new[:, k]))
        self.tau_ = tau_new.copy()
    

class LDA_FEM_base():
    def __init__(self):
        self.mean = None
        self.scatter = None
        self.cov = None

    def scatter_constant0(self, X):
        return 1/np.trace(self.scatter)
        
    def scatter_constant1(self, X):
        return X.shape[1]/np.trace(self.scatter)
    
    def scatter_constant2(self, X):
        diff = X - self.mean
        n = np.trace(self.scatter) / X.shape[1]
        s = self.scatter / n
        maha = (np.dot(diff, np.linalg.inv(s)) * diff).sum(1)
        #plt.plot(np.sort(maha))
        maha_avg = np.median(np.sort(maha)[:math.floor(0.8*len(maha))])
        #print("maha avg", maha_avg)
        return maha_avg / n / X.shape[1]
    
    def scatter_constant3(self, X):
        return self.scatter_constant2(X) * 1e6
    
    def FEM_estimate(self, X, normalisation_method=2): #X is {X: I(Zi=k|Xi=xi)=1}
        # initialise K=2 (for 1 class, but to maintain structure), tau,alpha randomly and means and sigma with Gaussian MLE
        _K = 2
        FEM_estimator = FEM_classification(_K, rand_initialization=True)
        FEM_estimator._initialize(X)
        FEM_estimator.override_params(np.repeat(X.mean(axis=0)[None,:], _K, axis=0), np.repeat(np.cov(X.T)[None,:], _K, axis=0))
        
        #run E-step to get indicators
        cond_prob = FEM_estimator._e_step_indicator(X)
        #run M-step
        FEM_estimation = FEM_estimator._m_step(X, cond_prob)  
        self.mean = FEM_estimation[1][0,:]; self.scatter = FEM_estimation[2][0,:,:]
        if normalisation_method == 1:
            s = self.scatter_constant1(X)
        elif normalisation_method == 0:
            s = self.scatter_constant0(X)            
        elif normalisation_method == 2:
            s = self.scatter_constant2(X)
        elif normalisation_method == 3:
            s = self.scatter_constant3(X)
        else:
            assert(1==0)
        self.cov = self.scatter * s
        #print("scatter->cov:", s)
        return [self.mean, self.cov]

class LDA_FEM(LDA, LDA_FEM_base):
    def __init__(self, method='distributional'):
        super().__init__(method)
        self.normalisation_method = 2
    def estimate_parameters(self, X):
        return self.FEM_estimate(X, self.normalisation_method)

class QDA_FEM(LDA_FEM, LDA_FEM_base):
    def __init__(self, method='distributional'):
        super().__init__(method)
        self.pool_covs = False
        
class t_LDA_FEM(t_LDA, LDA_FEM_base):
    def __init__(self, method='distributional'):
        super().__init__(method)
        self.normalisation_method = 2
    def estimate_parameters(self, X):
        params = self.FEM_estimate(X, self.normalisation_method)
        return params + [fit_t_dof(X, *params, dof_0=3)]
    
class t_QDA_FEM(t_LDA_FEM, LDA_FEM_base):
    def __init__(self, method='distributional'):
        super().__init__(method)
        self.pool_covs = False   
        
class FEMDA(QDA_FEM):
    def __init__(self, method='distributional'):
        super().__init__(method)
    def _posteriors2(self, X): #->KxN
        #print("hello")
        FEM = FEM_classification(self._K, rand_initialization=True)
        FEM._initialize(X)
        FEM.override_params(self.means_.T, self.covariance_)
        FEM.update_tau(X)
        #print(FEM.K)
        cond_prob = FEM._e_step(X)
        return cond_prob.T
    
    def simple_mahalanobis(self, X, m, S):
        diff = X - m
        return (np.dot(diff, np.linalg.inv(S)) * diff).sum(1)
    
    def _posteriors(self, X):
        N,p = X.shape
        log_maha = np.zeros((self._K, N))
        for k in range(self._K):
            log_maha[k, :] = np.log(self.simple_mahalanobis(X, self.means_[:,k], self.covariance_[k,:,:]))
        _,logdets = np.linalg.slogdet(self.covariance_)
        pik = -0.5 * (p * log_maha + logdets[:,None])
        return pik

class FEMDA_N(FEMDA):
    def __init__(self, method='distributional'):
        super().__init__(method)
    def fit(self, X,y):
        X_n = normalise_centered(X, y, mean_estimator=lambda x:LDA_FEM_base().FEM_estimate(x, self.normalisation_method))
        super().fit(X_n,y)

class FEM_predictor(FEM_classification):
    def __init__(self, K):
        super().__init__(K, rand_initialization=True)
    
    def predict(self, X_new, DA_means, DA_covs):
        assert(self.K == DA_covs.shape[0])
        self._initialize(X_new)
        self.override_params(DA_means.T, DA_covs)

        return super().predict(X_new, thres=0)

