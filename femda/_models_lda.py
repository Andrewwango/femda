import numpy as np
import time

from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ._algo_utils import label_outliers

class LDA(BaseEstimator, ClassifierMixin):
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

    def _dk_from_method(self, X): #NxM -> KxN
        if not(type(self.method) is str and self.method in ['generalised', 'distributional']):
            raise ValueError('Method must be generalised or distributional')
        if self.method=='generalised':
            return self._general_discriminants(X)
        elif self.method=='distributional':
            return self._log_likelihoods(X)
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        X, y = self._validate_data(X, y, ensure_min_samples=2, estimator=self,
                                   dtype=[np.float64, np.float32], ensure_min_features=2)

        st=time.time()

        self.classes_ = unique_labels(y) #1xK
        self._K = len(self.classes_)
        self._M = X.shape[1]
        self.X_classes_ = [X[np.where(y == k), :][0,:,:] for k in self.classes_] #Kxn_kxM
        n = np.array([c.shape[0] for c in self.X_classes_])
        
        self.priors_ = n / n.sum()
        #print(self.priors, n)
        
        try:
            self.parameters_ = [self._estimate_parameters(c) for c in self.X_classes_]
        except np.linalg.LinAlgError:
            self.parameters_ = [[np.zeros(self._M), np.eye(self._M)] for c in self.X_classes_]

        self.means_ = np.array([param[0] for param in self.parameters_]).T
        self.covariance_ = np.array([param[1] for param in self.parameters_])
        self.covariance_ = np.repeat(np.sum(n[:,None,None] * self.covariance_, axis=0)[None,:],self._K,axis=0) / n.sum() \
                if self.pool_covs else self.covariance_ 
                    


        assert(n.sum() == X.shape[0])
        assert(self._M == self.covariance_.shape[2])
        assert (np.allclose(self.priors_.sum(), 1))
        #print("Fitting time", time.time()-st)
        return self

    def _decision_function(self, X):
        check_is_fitted(self, ["means_", "covariance_", "priors_", "parameters_"])
        X = check_array(X)

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
        return dk.T #return in standard sklearn shape NxK

    def decision_function(self, X):
        dk = self._decision_function(X)
        if len(self.classes_) == 2:
            return dk[:,1] - dk[:,0]
        return dk #NxK

    def predict(self, X, percent_outliers=0):
        dk = self._decision_function(X)
        preds = np.nanargmax(dk, axis=1)# if len(self.classes_) != 2 else (dk > 0).astype(np.uint8)
        y = self.classes_[preds]
        return label_outliers(X, y, self.means_, self.covariance_, thres=percent_outliers)
       
    def predict_proba(self, X):
        dk = self._decision_function(X)
        likelihood = np.exp(dk - dk.max(axis=1)[:, np.newaxis])
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]
    

## Custom QDA
class QDA(LDA):
    def __init__(self, method='distributional'):
        super().__init__(method=method, pool_covs=False)