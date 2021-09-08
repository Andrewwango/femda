"""
Base models and related models for our robust FEMDA:
Flexible EM-Inspired Discriminant Analysis.
"""
import numpy as np
from math import sqrt

from ._algo_utils import regularize, get_reg_lambd, fit_t_dof, fit_gaussian
from ._models_lda import LDA
from ._models_t_lda import t_LDA


class _FEM_base():
    """Class to use the EM steps contained in the FEM clustering algorithm for
    parameter estimation of class labelled data. The EM steps here are 
    modified from the original clustering steps: they only take one class, and
    the E-step is deterministic since this is now a supervised algorithm.
    """
    def _e_step_indicator(self, X):
        ''' Pseudo E-step of clustering algorithm where all conditional 
        probabilities are replaced by ones according to the supervised labels.
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            data
    
        Returns
        ----------
        indicator_matrix: ndarray of shape (n_samples,)
             Matrix representing determined "class conditional probabilities".
             Single row as we only are concerned with one class.
        '''
        return np.ones((X.shape[0]))

    def _m_step(self, X, cond_prob, max_iter = 20, eps=1e-6):
        ''' M-step of clustering algorithm used to estimate parameters given
        conditional probabilities.
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            data
        cond_prob: array-like of shape (n_samples,)
            Conditional probability matrix from E-step, for one class only, 
            i.e (cond_prob)_ik = P(Z_i=k|X_i=x_i) where k = 0
        max_iter: int, default=20
            Max number of fixed point iterations
        eps: float, default=1e-6
            Convergence tolerance
    
        Returns
        ----------
        mean: ndarray of shape (n_features,)
            The new mean of each mixture component.
        sigma: ndarray of shape (n_features, n_features)
            The new regularized covariance of each mixture component.
        '''
        n, m = X.shape
        mean, sigma = fit_gaussian(X)
        convergence_fp      = False
        ite_fp              = 1
        safety = lambda x : np.minimum(0.5, x)
        while (not convergence_fp) and ite_fp<max_iter:
            ite_fp += 1

            sigma_inv = np.linalg.inv(regularize(sigma))
            diff = X - mean
            sq_maha = (np.dot(diff, sigma_inv) * diff).sum(1)     
            
            mean_new = np.dot(safety(cond_prob / sq_maha), X) \
                    / (safety(cond_prob / sq_maha).sum() + get_reg_lambd())

            sigma_new = np.dot(safety(cond_prob / sq_maha) * diff.T, diff) \
                    / (n + get_reg_lambd()) * m

            convergence_fp = sqrt(((mean - mean_new)**2).sum() / m) < eps \
                    and np.linalg.norm(sigma_new - sigma, ord='fro') / m < eps

            mean  = mean_new.copy()
            sigma = sigma_new.copy()

        return mean, regularize(sigma)

    def _estimate_parameters_with_FEM(self, X):
        """Estimate parameters (mean, scatter) of one class with FEM algorithm,
        according to flexible Elliptically Symmetrical model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data of one class (i.e. {X: I(Zi=k|Xi=xi)=1})

        Returns
        -------
        params : list of [ndarray of shape (n_features,),
                ndarray of shape (n_features, n_features)]\
            Estimated mean vector and covariance matrix.
        """

        cond_prob = self._e_step_indicator(X)

        #run M-step
        mean, cov = self._m_step(X, cond_prob)
        return mean, cov * X.shape[1] / np.trace(cov)


class LDA_FEM(LDA, _FEM_base):
    """Linear Discriminant Analysis with FEM-Inspired parameter estimation.
    Stepping stone to FEMDA! See `_models_LDA.LDA`, `_FEM_base` for more.
    Inherits from `_models_LDA.LDA` and `_FEM_base`.
    """        
    def _estimate_parameters(self, X):
        """Estimate parameters (mean, scatter) of one class with FEM algorithm,
        according to flexible Elliptically Symmetrical model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data of one class (i.e. {X: I(Zi=k|Xi=xi)=1})

        Returns
        -------
        params : list of [ndarray of shape (n_features,),
                ndarray of shape (n_features, n_features)]\
            Estimated mean vector and covariance matrix.
        """

        return self._estimate_parameters_with_FEM(X)

class QDA_FEM(LDA_FEM):
    """Quadratic Discriminant Analysis with FEM-Inspired parameter estimation.
    Stepping stone to FEMDA! See `LDA_FEM` for more.
    Inherits from LDA_FEM and unsets covariance pooling. 
    """
    def __init__(self, method='distributional'):
        super().__init__(method=method, pool_covs=False)
        
class t_LDA_FEM(t_LDA, _FEM_base):
    """Student-t Linear Discriminant Analysis with FEM-Inspired parameter
    estimation (discriminant function as usual).
    """
    def _estimate_parameters(self, X):
        """Estimate parameters (mean, scatter) of one class with FEM algorithm,
        according to flexible Elliptically Symmetrical model, along with usual
        degrees of freedom.
        """
        params = self._estimate_parameters_with_FEM(X)
        return params + [fit_t_dof(X, *params, dof_0=3)]
    
class t_QDA_FEM(t_LDA_FEM):
    """Student-t Quadratic Discriminant Analysis with FEM-Inspired parameter
    estimation (discriminant function as usual).
    """
    def __init__(self, method='distributional'):
        super().__init__(method=method, pool_covs=False)
        

class FEMDA(QDA_FEM):
    """FEM-Inspired Discriminant Analysis implementation.
    FEM-Inspired parameter estimation along with FEM E_step discrimination.
    See `QDA_FEM` for more. Inherits from `QDA_FEM`.
    """
    def __init__(self):
        super().__init__(method='distributional')
    
    def _simple_mahalanobis(self, X, m, S):
        """Calculate Mahalanobis distance of data points.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        m : array-like of shape (n_features,)
            Mean vector.
        S : array-like of shape (n_features, n_features)
            Scatter matrix.

        Returns
        -------
        d : ndarray of shape (n_samples,)
            Mahalanobis distances.
        """
        diff = X - m
        return (np.dot(diff, np.linalg.inv(S)) * diff).sum(axis=1)
    
    def _log_likelihoods(self, X):
        """Calculate log likelihood of new data per class according to Flexible
        Elliptically Symmetrical model using FEM E-step implementation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.

        Returns
        -------
        p : ndarray of shape (n_classes, n_samples)
            Log likelihood per class of new data.
        """
        N,p = X.shape
        log_maha = np.zeros((self._K, N))
        for k in range(self._K):
            log_maha[k, :] = np.log(self._simple_mahalanobis(X, 
                                    self.means_[:,k], self.covariance_[k,:,:]))
        
        _,logdets = np.linalg.slogdet(self.covariance_)
        pik = -0.5 * (p * log_maha + logdets[:,None])
        return pik


class FEMDA_N(FEMDA):
    """Experimental FEM-Inspired Discriminant Analysis implementation, but
    with pre-normalisation of data. This should produce similar results,
    since the FEMDA model fits an arbitrary scale parameter.
    See `FEMDA` for more.
    """
    def _normalise_centered(self, X, y, mean_estimator=fit_gaussian):
        """Estimate means and normalise data based around them.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.
        
        mean_estimator : func, default=`_algo_utils.fit_gaussian`
            Fucnction which returns list of parameters, the first element of
            which returns a mean vector.
        """        
        def _normalise(a):
            return (a.T / np.linalg.norm(a, axis=1)).T
        X_copy = X.copy()
        for k in np.unique(y):
            mean = mean_estimator(X[y==k])[0]
            X_copy[y==k] = _normalise(X[y==k] - mean) + mean
        return X_copy

    def fit(self, X, y):
        """Fit pre-normalised FEMDA model according to data. The data are 
        pre-normalised using the means pre-estimated with FEM estimation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.
        """
        X_n = self._normalise_centered(X, y, mean_estimator = 
                    lambda x:_FEM_base()._estimate_parameters_with_FEM(x))
        print("X", X)
        print("X-N", X_n)
        return super().fit(X_n,y)

