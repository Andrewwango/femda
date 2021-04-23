import numpy as np
import random 
import pandas as pd
import os
import time
import csv 
import warnings
from dataset_utils import *
from fit import *

# MATH and STATS:
import math
from scipy import stats, special, optimize, spatial

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
r = robjects.r
psych = importr('psych')
rrcov = importr('rrcov')
SpatialNP = importr('SpatialNP')
LaplacesDemon = importr('LaplacesDemon')



## Custom LDA with Gaussians
class LDA():
    def __init__(self, method='distributional'):
        self.method = method
        self.K = None
        self.M = None
        self.ks = None #1xK
        self.priors = None #1xK
        self.coefficients = None #KxM
        self.intercepts = None #1xK
        self.parameters = None
        self.means = None #MxK
        self.covariances =  None #KxMxM
        self.pool_covs = True
        self.fudge = 1
    
    def _discriminants(self, X): #NxM -> KxN
        if (self.coefficients is None) or (self.intercepts is None):
            self.calculate_discriminant_params()
        assert((self.priors is not None) and (self.coefficients is not None) and (self.intercepts is not None))
        return self.coefficients.dot(X.T) + (self.intercepts + np.log(self.priors))[None,:].T
    
    def _bose_k(self):
        return np.array([0.5])
    
    def _mahalanobis(self, X, ki=None): #NxM -> KxN
        ret = []
        r = range(self.K) if ki is None else [ki]
        for k in r:
            m = X - self.means[:,k]
            kth_maha = np.array(list(map(lambda d: d @ np.linalg.inv(self.covariances)[k,:,:] @ d[:,None], m))).T
            #kth_maha = np.diag(m @ np.linalg.inv(self.covariances)[k,:,:] @ m.T)]
            ret += [kth_maha]
        return np.vstack(ret) if ki is None else ret[0]
    
    def _general_discriminants(self, X): #KxN
        return -0.5*np.log(np.linalg.det(self.covariances))[:,None] * self.fudge - self._bose_k()[:,None] * self._mahalanobis(X)
    
    def _kth_likelihood(self, k):
        return stats.multivariate_normal(mean=self.means[:,k], cov=self.covariances[k,:,:])
    
    def _posteriors(self, X):
        r = [self._kth_likelihood(k).pdf(X) for k in range(self.means.shape[1])] #times priors!
        return np.array(r)
       
    def estimate_parameters(self, X): #NxM -> [1xM, MxM]
        return [X.mean(axis=0), np.cov(X.T)]
    
    def calculate_discriminant_params(self):
        cov_inv = np.linalg.inv(self.covariances[0,:,:])
        self.coefficients = self.means.T.dot(cov_inv)
        self.intercepts = -0.5*np.diag(self.means.T.dot(cov_inv.dot(self.means)))  
    
    def fit(self, X, y):
        st=time.time()
        self.ks = np.unique(y); self.K = len(self.ks); self.M = X.shape[1]
        classes = [X[np.where(y == k), :][0,:,:] for k in self.ks] #Kxn_kxM
        n = np.array([c.shape[0] for c in classes])
        self.parameters = [self.estimate_parameters(c) for c in classes]
        self.means = np.array([param[0] for param in self.parameters]).T
        self.covariances = np.array([param[1] for param in self.parameters])
        self.covariances = np.repeat(np.sum(n[:,None,None] * self.covariances, axis=0)[None,:],self.K,axis=0) / n.sum() \
                if self.pool_covs else self.covariances 
                    
        self.priors = n / n.sum()
        #print(self.priors.sum())
        assert(n.sum() == X.shape[0])
        assert(self.M == self.covariances.shape[2])
        assert (np.allclose(self.priors.sum(), 1))
        #print("Fitting time", time.time()-st)
        return classes
    
    def _dk_from_method(self, X):
        if self.method=='distributional':
            return self._posteriors(X)
        elif self.method=='coeffs':
            return self._discriminants(X)
        elif self.method=='generalised':
            return self._general_discriminants(X)
                
    def predict(self, X, percent_outliers=0):
        dk = self._dk_from_method(X)
        y = self.ks[dk.argmax(axis=0)]
        return label_outliers(X, y, self.means, self.covariances, thres=percent_outliers)
       
    def predict_proba(self, X):
        dk = self._dk_from_method(X)
        if self.method!='distributional':
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
        return stats.multivariate_t(loc=self.means[:,k], shape=self.covariances[k,:,:], df=self.dofs[k])
    
    def estimate_parameters(self, X):
        return fit_t(X)
    
    def _bose_k(self):
        return (0.5*(1 + self.M/self.dofs))
    
    def _discriminants(self, X): #NxM -> KxN
        return None
        
    def _general_discriminants(self, X):
        v = self.dofs
        return super()._general_discriminants(X) + (special.gammaln((v+self.M)/2) - special.gammaln(v/2) - 0.5*self.M*np.log(v))[:,None]
            
    def fit(self, X,y):
        super().fit(X,y)
        self.dofs = np.array([param[2] for param in self.parameters]).squeeze()

"""def _discriminants(self, X): #NxM -> KxN
    if (self.coefficients is None) or (self.intercepts is None):
        self.calculate_discriminant_params()
    assert((self.priors is not None) and (self.coefficients is not None) and (self.intercepts is not None))
    v = self.dofs
    cov_inv = np.linalg.inv(self.covariances[0,:,:])
    return self.coefficients.dot(X.T) + (self.intercepts + np.log(self.priors))[None,:].T + np.outer(np.diag(X.dot(cov_inv.dot(X.T))), -0.5*(1+(self.M/v)).reshape(1,self.K)).T
def calculate_discriminant_params(self): #obsolete
    cov_inv = np.linalg.inv(self.covariances[0,:,:])
    v = self.dofs
    self.coefficients = self.means.T.dot(cov_inv) * (1+(self.M/v)).reshape(self.K,1)
    self.intercepts = np.diag(self.means.T.dot(cov_inv.dot(self.means))) * -0.5*(1+(self.M/v)).reshape(1,self.K) + \
        (special.gammaln((v+self.M)/2) - special.gammaln(v/2) - 0.5*self.M*np.log(v)).reshape(1,self.K)
    self.intercepts = self.intercepts.squeeze()
Note: t-distribution discriminants can't be linear as c * xTSx depends on v (and k) in general"""

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
        classes = super().fit(X,y) #Kx[n_k, M]
        
        if c is not None:
            self.c = c
            return 
            
        uijs = [np.zeros((classes[k].shape[0], self.K, self.K)) for k in range(self.K)] #Kx[n_kxIxJ]
        sij = np.zeros((self.K,self.K))
        logdets = np.log(np.linalg.det(self.covariances)) #K,  
        for i in range(self.K):
            for j in range(self.K):
                dij_on_i = self._mahalanobis(classes[i], ki=j) - self._mahalanobis(classes[i], ki=i) #Kxn_i
                dij_on_j = self._mahalanobis(classes[j], ki=j) - self._mahalanobis(classes[j], ki=i) #Kxn_j
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
            
            for i in range(self.K):
                Rijc = []
                for j in range(self.K):
                    if i==j: continue
                    p = uijs[i][:, i,j]
                    to_app = p > -c if sij[i,j]>0 else p < -c 
                    Rijc.append(classes[i][to_app])
                Rijc = np.vstack(Rijc)
                Ric = np.unique(Rijc, axis=0)
                #print(Ric.shape, Rijc.shape)
                lenRic = Ric.shape[0]
                MCic = classes[i].shape[0] - lenRic
                #print(MCic, Ric.shape)
                MCc[e] += MCic
                
        #return uijs, MCc, T
        c_star = T[MCc.argmin()]
        self.c = c_star if c_star > 0 else 0.001
        print("optimal c is", c_star)
        
    def _bose_k(self):
        return np.array([0.5/self.c])
        
        #def predict(self, X):

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


def label_outliers_kth(X_k, mean, cov, thres=0.05):   
    outlierness = np.zeros((X_k.shape[0], )).astype(bool)         
    thres = stats.chi2.ppf(1 - thres, X_k.shape[1])

    diff = X_k - mean
    sig_cluster = np.mean(diff * diff) 
    maha_cluster = (np.dot(diff, np.linalg.inv(cov)) * diff).sum(1) / sig_cluster
    outlierness = (maha_cluster >  thres)
    return outlierness

def label_outliers_kth2(X_k, mean, cov, thres=0):
    diff = X_k - mean
    maha = (np.dot(diff, np.linalg.inv(cov)) * diff).sum(1)
    _,n_to_keep = split(X_k.shape[0], thres)
    t = maha[np.argsort(maha)[n_to_keep-1]]
    outlierness = (maha > t)
    return outlierness

def label_outliers(X,y, means,covs, thres=0.05):
    if thres==0: return y
    y_new = y.copy()
    ks = np.unique(y)
    for ki, k in enumerate(ks):
        k = int(k)
        outlierness = label_outliers_kth2(X[y==k,:], means[:,ki], covs[ki,:,:], thres=thres)
        #print(outlierness.sum())
        b = np.where(y==k)[0][outlierness]
        #print(b)
        y_new[b]=-1#k+5
    return y_new

def errors_means(true, pred):
    #print(true,pred)
    return np.array([(np.square(t-pred[i])).sum() for i,t in enumerate(true)])
    #return np.array([np.square(t-pred.T).sum(axis=1).min() for t in true.T])#.mean()

def errors_covs(true, pred):
    p = true[0].shape[0]
    assert (true[0].shape[1]==pred[0].shape[0])
    
    def error_cov(cov1, cov2): return np.linalg.norm(cov1/np.trace(cov1)*np.trace(cov2) - cov2, ord='fro')/(p*p)
    
    return np.array([error_cov(pred[i], t) for i,t in enumerate(true)])
    #return np.array([np.min([error_cov(pred_cov, true_cov) for pred_cov in pred]) for true_cov in true])


def evaluate_estimators(model, true_means, true_covs):
    means = model.means.T
    covs = model.covariances
    true_means_list = [true_means[key] for key in sorted(true_means.keys())]
    #print(means, true_means_list)
    true_covs_list = [true_covs[key] for key in sorted(true_covs.keys())]
    return errors_means(true_means_list, means), errors_covs(true_covs_list, covs)