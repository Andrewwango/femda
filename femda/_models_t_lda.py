import numpy as np

from scipy import stats, special

from ._algo_utils import fit_t
from ._models_lda import LDA

## t-LDA (mostly redundant: pooling covs still means discriminant depends on x^2 because of differing dofs (assuming same dofs = MMD classifier)
class t_LDA(LDA):   
    def _kth_likelihood(self, k):
        return stats.multivariate_t(loc=self.means_[:,k], shape=self.covariance_[k,:,:], df=self.dofs_[k])
    
    def _estimate_parameters(self, X):
        return fit_t(X)
    
    def _bose_k(self):
        return (0.5*(1 + self._M/self.dofs_))
        
    def _general_discriminants(self, X):
        v = self.dofs_
        return super()._general_discriminants(X) + (special.gammaln((v+self._M)/2) - special.gammaln(v/2) - 0.5*self._M*np.log(v))[:,None]
            
    def fit(self, X,y):
        super().fit(X,y)
        self.dofs_ = np.array([param[2] for param in self.parameters_]).squeeze() #1xK
        return self

## t-QDA
class t_QDA(t_LDA):
    def __init__(self, method='distributional'):
        super().__init__(method=method, pool_covs=False)