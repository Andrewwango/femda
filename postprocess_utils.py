import numpy as np

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