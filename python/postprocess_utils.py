import numpy as np
from scipy.optimize import linear_sum_assignment as linear_sum_assignment


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

def evaluate_all(models, true_means, true_covs, plot=True, ret=False):  
    models = models if type(models) is not dict else list(models.values())
    labels = [type(model).__name__ for model in models]
    all_estimator_errors = []
    for model in models:
        #print(type(model).__name__)
        all_estimator_errors.append(evaluate_estimators(model, true_means, true_covs))
    all_estimator_errors = np.array(all_estimator_errors)
    
    data_means_errors = {}
    data_covs_errors = {}
    for k in range(len(all_estimator_errors[0,0])):
        data_means_errors[str(k)] = all_estimator_errors[:,0,k]
        data_covs_errors[str(k)] = all_estimator_errors[:,1,k]
    
    if ret:
        resultats = np.zeros((len(models), 2))
        resultats[:,0] = np.median(np.vstack(data_means_errors.values()), axis=0)
        resultats[:,1] = np.median(np.vstack(data_covs_errors.values()), axis=0)
        return resultats
    
    #print(data_means_errors, data_covs_errors, labels)
    if plot:
        fig,(ax1,ax2) = plt.subplots(2,1)
        bar_plot(ax1, data_means_errors, labels)
        bar_plot(ax2, data_covs_errors, labels)

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind).T
    
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size