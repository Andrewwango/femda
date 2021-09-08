import numpy as np
from scipy import special, optimize
import math

def get_reg_lambd():
    """Return default regularization parameter.
    """
    return 1e-5

def regularize(sigma, lambd = get_reg_lambd()):    
    """Regularizes matrix to avoid singular matrix issues.

    Args:
        sigma (array-like of shape (n_features, n_features)): scatter matrix
        lambd (float, optional): reg parameter. Defaults to get_reg_lambd().

    Returns:
        array-like of shape (n_features, n_features):
            regularized scatter matrix
    """
    return sigma + np.eye(len(sigma)) * lambd

def fit_gaussian(X):
    """Fit Gaussian distribution parameters to data, with regularization.

    Args:
        X (array-like of shape (n_samples, n_features)): Training data.

    Returns:
        list of [ndarray of shape (n_features,),
                ndarray of shape (n_features, n_features)] :
            Estimated mean vector and covariance matrix.
    """
    n_reg = X.shape[0] + get_reg_lambd()
    m = (X.sum(axis=0) / n_reg)[None, :]
    return m[0], regularize(np.dot( (X-m).T, X-m ) / (n_reg - 1))

def t_EM_e_step(D, dof, mu, cov):
    """Run one E-step of the EM algorithm to fit Student-t. See Murphy,
    Machine Learning: A Probabilistic Perspective for details.

    Args:
        D (int): n_features of data
        dof (float): estimated degrees of freedom
        mu (array-like of shape (n_samples, n_features)): X - mean
        cov (array-like of shape (n_features, n_features)): estimated scatter

    Returns:
        z
        delta
        See Murphy for details.
    """
    delta = np.einsum('ij,ij->i', mu, np.linalg.solve(regularize(cov),mu.T).T)
    z = (dof + D) / (dof + delta)
    return z,delta

def fit_t_dof(X, mean, cov, dof_0, max_iter=200, mu=None, tol=1e-3):
    """Fit degrees of freedom to data according to Student-t, given other 
    parameters.

    Args:
        X (array-like of shape (n_samples, n_features)): Training data.
        mean (array-like of shape (n_features,)): Mean vector.
        cov (array-like of shape (n_features, n_features)): Scatter matrix.
        dof_0 (float): Initial guess of degrees of freedom.
        max_iter (int, optional): Max number of iterations. Defaults to 200.
        mu (array-like, optional): [description]. Defaults to None.
            X - mean. If set, skips this calculation.
        tol (float, optional): Convergence tolerance. Defaults to 1e-3.

    Returns:
        float: Estimated degree of freedom.
    """
    N, D = X.shape
    mu = mu if mu is not None else X - mean.squeeze()[None,:]
    dof = dof_0
    i = 0

    while i < max_iter:
        z,_ = t_EM_e_step(D, dof, mu, cov)

        d_t = (np.log(z) + special.digamma((dof + D) / 2)
                 - np.log((dof + D) / 2) - z).sum()
        dof_obj = lambda v: - (-N * special.gammaln(v/2)
                             + N * v * np.log(v/2) / 2 + v * d_t / 2 )
        dof_grad = lambda v: - (N / 2 * (-special.digamma(v/2)
                                        + np.log(v/2) + 1) + d_t/2)        
        dof_new = optimize.minimize(dof_obj, 
                                    dof, 
                                    jac=dof_grad,
                                    bounds=[(0, None)]
        ).x

        if abs(dof_new-dof)/dof <= tol: 
            dof = dof_new
            break
        dof = dof_new
        i += 1

    return dof
    

def fit_t(X, iter=20, eps=1e-6):
    """Fit Student-t distribution to data, according to EM-algorithm as
    described in Murphy, Machine Learning: A Probabilistic Perspective.
    Initialise with Gaussian MLE estimations.

    Args:
        X (array-like of shape (n_samples, n_features)): Training data.
        iter (int, optional): Max number of EM iterations. Defaults to 200.
        eps (float, optional): EM convergence tolerance. Defaults to 1e-6.

    Returns:
        list of [ndarray of shape (n_features,),
                ndarray of shape (n_features, n_features), float] :
            Estimated mean vector,  covariance matrix and degree of freedom.
    """
    N,D = X.shape
    mean, cov = fit_gaussian(X)
    mu = X - mean[None,:]
    dof = 3
    obj = []

    for i in range(iter):
        #print("mean at start", i, mean)
        if i>198: print("t not converged", obj[-1])
        # E step
        z,delta = t_EM_e_step(D, dof, mu, cov)
        
        obj.append(
            - N * np.linalg.slogdet(cov)[1]/2 - (z * delta).sum()/2 \
            - N * special.gammaln(dof/2) + N * dof*np.log(dof/2)/2 
            + dof * (np.log(z)-z).sum()/2)

        if len(obj) > 1 and np.abs(obj[-1] - obj[-2]) < eps: 
            break
        
        # M step
        mean = (X * z[:,None]).sum(axis=0).reshape(-1,1) / z.sum()
        #print("mean at end", i, mean)
        mu = X - mean.squeeze()[None,:]
        cov = np.einsum('ij,ik->jk', mu, mu * z[:,None])/N
        dof = fit_t_dof(X, None, cov, dof, max_iter=1, mu=mu)
    #plt.plot(obj)
    #plt.show()
    #print(obj)
    return mean.squeeze(), regularize(cov), dof

def fit_t2(X, eps = 1e-5, max_iter = 20):

    n, m          = X.shape
    n_clusters = n + 1e-5
    mean, covariance = fit_gaussian(X)
    v    = 1

    convergence      = False
    ite              = 1

    while (not convergence) and ite<max_iter:
        print("m at beginning of ", ite, mean)
        prev_mean = mean.copy()
        prev_cov = covariance.copy()
        prev_v = v

        ite              = ite + 1 
        mean             = np.zeros(m)
        covariance       = np.zeros([m,m])
        sum_mean_weights = 1e-5
        sum_nu_weights   = 1e-5
        for i in range(n):
            mean_weight      = (v + m) / (v + np.dot(np.array([X[i]-mean]), np.dot(np.linalg.inv(regularize(covariance)), np.array([X[i]-mean]).T))[0][0])
            
            sum_mean_weights += mean_weight
            covariance += np.dot(np.array([X[i]-mean]).T, np.array([X[i]-mean])) * mean_weight
            mean             += mean_weight * X[i]
            sum_nu_weights  += (np.log(mean_weight) - mean_weight - np.log((v + m)/2) + special.digamma((v + m)/2)) / n_clusters
        def f(nu):
            return special.gammaln(nu/2) - 0.5*nu*np.log(nu/2) - 0.5*nu*sum_nu_weights
        def grad_f(nu):
            return 0.5*special.digamma(nu/2) - 0.5*np.log(nu/2) - 0.5 - 0.5*sum_nu_weights
        v = optimize.minimize(f, v, jac = grad_f,bounds=[(0,None)]).x[0]
        
        mean /= sum_mean_weights
        covariance /= n_clusters
        print("m at end of ", ite, mean)
        convergence = abs(v-prev_v) + sum(abs(mean-prev_mean)) + sum(sum(abs(covariance-prev_cov)))
        #print(convergence, "at step", ite)
        convergence = False#convergence < eps
    
    return mean, regularize(covariance), v



def label_outliers_kth2(X_k, mean, cov, thres=0):
    """Label outliers for kth class according to Mahalanobis distance.

    Args:
        X_k (array-like of shape (n_samples, n_features)): Training data for
            kth class.
        mean (array-like of shape (n_features,)): Mean vector.
        cov (array-like of shape (n_features, n_features)): Scatter matrix.
        thres (float, optional): Mahalanobis outlier threshold. Defaults to 0.

    Returns:
        array-like of type bool, shape (n_samples,) : whether samples are 
            outliers.
    """
    diff = X_k - mean
    maha = (np.dot(diff, np.linalg.inv(cov)) * diff).sum(1)
    def split(n, perc):
        a = int(np.floor(n *perc))
        return a, n-a
    _,n_to_keep = split(X_k.shape[0], thres)
    t = maha[np.argsort(maha)[n_to_keep - 1]]
    outlierness = (maha > t)
    return outlierness

def label_outliers(X, y, means, covs, thres=0.05):
    """Label outliers in data according to Mahalanobis distance.

    Args:
        X_k (array-like of shape (n_samples, n_features)): Training data for
            kth class.
        means (array-like of shape (n_features, n_classes)): Means vectors.
        covs (array-like of shape (n_classes, n_features, n_features)): 
            Scatter matrices.
        thres (float, optional): Mahalanobis threshold. Defaults to 0.05.

    Returns:
        array-like (shape (n_samples,)) : preds with outliers relabelled as -1.
    """
    if thres == 0:
        return y

    y_new = y.copy()
    ks = np.unique(y)

    for ki, k in enumerate(ks):
        k = int(k)
        outlierness = label_outliers_kth2(X[y==k,:], means[:,ki], covs[ki,:,:], 
                                          thres=thres)
        b = np.where(y==k)[0][outlierness]
        y_new[b] = -1#k+5

    return y_new

