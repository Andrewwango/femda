import random, math
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.optimize import linear_sum_assignment as linear_sum_assignment
from sklearn import decomposition #PCA
from sklearn import metrics # AMII and ARI
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from plotnine import ggplot, aes
from plotnine.data import *
from femda import *
def import_subset(dataset, labels, selected_cat, n_pca, n_sample, pca=None):
    
    i = 1
    assert (n_pca > 0 or n_pca == -1)
    
    if n_sample != -1:
        # n_sample from each category
        subset_labels = np.zeros((n_sample*len(selected_cat), ))  

        data_label = dataset.loc[selected_cat[0]==labels]
        sample = random.sample(range(data_label.shape[0]), n_sample)
        subset_data = data_label.iloc[sample, :]
        subset_labels[(n_sample*(i-1)):(n_sample*i)] = selected_cat[0]    

        for cat in selected_cat[1:]:
            i += 1
            data_label = dataset.loc[labels==cat]
            sample = random.sample(range(data_label.shape[0]), n_sample)
            subset_data = pd.concat([subset_data, data_label.iloc[sample, :]])
            subset_labels[(n_sample*(i-1)):(n_sample*i)] = cat  
    else:
        subset_data = dataset.copy()
        subset_labels = labels.copy()
    
    if pca is None and n_pca != -1:
        pca = decomposition.PCA(n_components = n_pca)
        pca.fit(subset_data)
    
    return_data = np.array(subset_data) if n_pca == -1 else pca.transform(subset_data)

    return return_data, np.array(subset_labels).astype(int), subset_data, None, pca


def split(n, perc):
    a = int(np.floor(n*perc))
    #b = int(np.ceil(n*(1-perc)))
    b = n-a
    #print(a,b,n)
    assert (a+b==n)
    return a,b

def apply_outlierness(outliernesses, data):
    y = data.copy()
    outlierness = np.hstack(outliernesses)
    #print(outlierness.shape, y.shape)
    assert(len(outlierness)==y.shape[0])
    y[outlierness] = -1
    return y

def contaminated(n, dist, contamination, loc, shape, df, hard):
    a,b = split(n, contamination)
    X1 = dist(loc=loc, shape=shape, df=df).rvs(size=b)
    if contamination == 0:
        return X1, np.zeros((X1.shape[0]))>1
    else:
        if not hard:
            X1c = dist(loc=loc, shape=4*shape, df=df).rvs(size=a)
            return np.vstack([X1, X1c]), np.hstack([np.zeros((b))>1, np.ones((a))>0])
        else:
            X1cs = np.zeros((a, shape.shape[0]))
            taus = np.linspace(1e-3, 1e4, a)
            for i,tau in enumerate(taus):
                X1cs[i, :] = dist(loc=loc, shape=tau*shape, df=df).rvs(size=1)
            return np.vstack([X1, X1cs]), np.hstack([np.zeros((b))>1, np.ones((a))>0])


def combine_dataset(X1, X2, X1perc):
    assert(X1.shape[0] == X2.shape[0])
    a,b = split(X1.shape[0], X1perc)
    out = np.hstack([X1[:a], X2[a:]]) if X1.ndim == 1 else np.vstack([X1[:a,:], X2[a:,:]])
    #np.random.shuffle(out)
    return out
    
def contaminate_dataset(a, perc):
    c = int(np.floor(len(a)*perc))
    idx = np.random.choice(len(a),c,replace=False)
    taus = np.linspace(1e-3, 1e4, c)
    a[idx, :] = taus[:,None]*a[idx, :]
    outlierness = np.zeros((a.shape[0]))
    outlierness[idx] = 1
    return a, outlierness>0

def mislabelled(n, mislabelling, labels):
    a,b = split(n, mislabelling)
    return np.hstack([np.random.permutation(np.hstack([i*np.ones((b)), labels[np.random.randint(0, len(labels), (a))]])) for i in labels])

def mislabelled_irregular(a, mislabelling):
    labels = a.copy()
    u = np.unique(a)
    idx = np.random.choice(len(labels),int(np.floor(len(labels)*mislabelling)),replace=False)
    labels[idx] = np.random.choice(u, len(labels[idx]), replace=True)
    return labels

def flip_bits(a, perc):
    idx = np.random.choice(len(a),int(np.floor(len(a)*perc)),replace=False)
    a[idx] = 1-a[idx]
    return a

def toeplitz(r, p):
    return linalg.toeplitz(np.array([r**j for j in range(p)]))

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
    """
    if plot:
        fig,(ax1,ax2) = plt.subplots(2,1)
        bar_plot(ax1, data_means_errors, labels)
        bar_plot(ax2, data_covs_errors, labels)
    """

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

def lda_grid(em, mapper):
    AA, BB = np.meshgrid(np.linspace(em.A.min(), em.A.max(), 50),
                     np.linspace(em.B.min(), em.B.max(), 50))
    return AA, BB, mapper.inverse_transform(np.c_[AA.ravel(), BB.ravel()])

def plot_contours_UMAP(gg, lda, AA, BB, grid):
    print(grid.shape)
    cmaps = ['Reds','Greens','Blues','Purples','Oranges','Greys','YlOrRd','PuBu','YlGn','pink'] 
    cnt = lda.predict_proba(grid)
    fig = gg.draw()
    ax = fig.get_axes()[0]
    for i in range(cnt.shape[1]):
        ax.contour(AA, BB, cnt[:,i].reshape(AA.shape), cmap=cmaps[i])#, [0.5])
        
def plot_regions_UMAP(gg, lda, AA, BB, grid):
    colors = ['r','g','b','c','m','y','k','chartreuse','orange','purple']
    preds = lda.predict(grid)
    #cnt = lda.predict_proba(grid)
    u = np.unique(preds)
    fig = gg.draw()
    ax = fig.get_axes()[0]
    grid_size = 50
    #pix_size = 
    plt.imshow(preds.reshape(grid_size,-1), extent=[AA.min(),AA.max(),BB.max(),BB.min()],cmap='gray')
    return
    for i in range(grid_size):
        for j in range(grid_size):
            print(colors[np.where(preds[i*grid_size+j] == u)[0][0]])
            ax.scatter(AA[i,j], BB[i,j], marker='s', ms=2, c=colors[np.where(preds[i*grid_size+j] == u)[0][0]])

def print_metrics(true, pred, conf=False, verbose=True, ret=False):
    dp = 5
    if true is None or pred is None:
        return (0,0,0)
    true = np.array(true)
    pred = np.array(pred)
    t = true#[pred>-1]
    p = pred#[pred>-1]
    #print(t,p)
    label = t.astype(int)
    accuracy = round(acc(label, p.astype(int)), dp)
    ari = round(metrics.adjusted_rand_score(label, p.astype(str)), dp)
    ami = round(metrics.adjusted_mutual_info_score(label, p.astype(str)), dp)
    if verbose:
        print("N", len(t), "Accuracy", accuracy, "ARI", ari, "AMI", ami)
        if conf:
            print(metrics.confusion_matrix(t.astype(int), p.astype(int)))
    
    if ret:
        return (accuracy, ari, ami)
    
    
    
def plot_contours(X, f, ax, lims=None):
    lims = [[X[:,0].min(), X[:,0].max()],[X[:,1].min(), X[:,1].max()]] if lims is None else lims
    AA, BB = np.meshgrid(np.linspace(*(lims[0]), 50),
                 np.linspace(*(lims[1]), 50))
    cmaps = ['Reds','Greens','Blues','Purples','Oranges','Greys','YlOrRd','PuBu','YlGn','pink'] 
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    grid = np.c_[AA.ravel(), BB.ravel()]
    cnt = f(grid)
    for i in range(cnt.shape[1]):
        #ax.contour(AA, BB, cnt[:,i].reshape(AA.shape), cmap=cmaps[i])#, [0.5])
        ax.contour(AA, BB, cnt[:,i].reshape(AA.shape), levels=[0.5], linewidths=3)#, colors=colors[i])#, [0.5])
    return cnt

def plot_dataset(X, y, ax, lims=None):
    for u in np.unique(y):
        ax.scatter(X[y==u,0],X[y==u,1], label=str(u))
        if lims is not None:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])

def plot_models(X, y, X_test, models, percent_outliers=0, lims=[[-10,10],[-10,10]]):
    models = models if type(models) is not dict else list(models.values())
    many = (type(models) is tuple or type(models) is list) and len(models)>1
    models = models if many else [models]
    cols = 5
    rows = 2*math.ceil(len(models)/cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*4,rows*4)) if many else plt.subplots(1,2, figsize=(8,4))
    axs = axs if many else axs[:,None]
    for i,model in enumerate(models):
        if many:
            zero = int(i/cols)*2; one = int(i/cols)*2+1; i = i % cols
        else:
            zero=0; one=1
        for a in [zero, one]:
            axs[a,i].tick_params(labelsize='small', length=0, pad=-10)
            axs[a,i].set_xlim(lims[0]); axs[a,i].set_ylim(lims[1])
        axs[zero,i].set_title(type(model).__name__, y=0.9)
        plot_dataset(X, y, axs[zero,i])
        plot_contours(X, model.predict_proba, axs[zero,i], lims)
        plot_dataset(X_test, model.predict(X_test, percent_outliers=percent_outliers), axs[one,i])
    plt.subplots_adjust(wspace=0, hspace=0)

def plot_means(models):
    for i,model in enumerate(models):
        means = model.means + (np.random.random_sample(size=model.means.shape)-0.5)*0.5
        plt.scatter(means[0,:], means[1,:], label=type(model).__name__)
    plt.legend()

    

def box_plot(result_book, algos=None, measures=None, e_loc_lims=None, e_sc_lims=None):
    
    # result_book: [algos, measures, runs]
    sbsize = (2,3) if result_book.shape[1] == 6 else (2,2)
    #if 
    sbmap = {0: [1,0], 1: [0,0], 2:[0,1], 3:[0,2], 4:[1,1], 5:[1,2]} if result_book.shape[1] == 6 else {0: [1,1], 1: [0,0], 2:[0,1], 3:[1,0]}
    fig,axs = plt.subplots(*sbsize, figsize=(4*len(measures)/2,4*2))
    
    for i in range(len(measures)):
        m,n = sbmap[i]
        axs[m,n].boxplot(result_book[:,i,:].T)
        axs[m,n].set_xticklabels(algos, rotation=45)
        axs[m,n].set_ylabel(measures[i])
        #if "estimation" in measures[m]:
            #femdaloc = np.where(["FEMDA" in k for k in algos])[0][0]
            #axs[m].set_ylim(bottom=-0.00001, top=result_book[femdaloc, m, :].max()*100)
    
    if e_loc_lims is not None:
        i = np.where(["estimation" in k for k in measures])[0][0]
        m,n = sbmap[i]
        axs[m,n].set_ylim(e_loc_lims)
    if e_sc_lims is not None:
        i = np.where(["estimation" in k for k in measures])[0][1]
        m,n = sbmap[i]
        axs[m,n].set_ylim(e_sc_lims)    
    fig.tight_layout(pad=2.0)
    extents = []
    for i in range(len(measures)):
        m,n = sbmap[i]
        extents.append(full_extent(axs[m,n]).transformed(fig.dpi_scale_trans.inverted()))
    return fig, extents

def full_extent(ax, pad=0.05):
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() #+ ax.get_yticklabels()
    items += [ax] #ax.get_xaxis().get_label()
    items += [ax.get_yaxis().get_label()]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

def save_subplots(fn, figure, extents, folder='images', extent_expansion=1, ext='pdf'):
    for i,extent in enumerate(extents):
        figure.savefig('{0}/{1}{2}.{3}'.format(folder, fn, i, ext), bbox_inches=extent.expanded(extent_expansion, extent_expansion))
    
def plot_tQDA_vs_FEMDA(result_book, xaxis=None, ylim=None):
    fig,ax = plt.subplots()
    xaxis = xaxis if xaxis is not None else np.arange(result_book.shape[2])
    ax.plot(xaxis, result_book[-3,1,:], label='tQDA')
    ax.plot(xaxis, result_book[-2,1,:], label='femda')
    ax.set_ylim(ylim)
    ax.set_xlabel("n_pca")
    ax.set_ylabel("Accuracy")
    ax.legend()
    
def bar_plot(ax, data, labels, colors=None, total_width=0.8, single_width=1, legend=True):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """

    # Check if colors where provided, otherwhise use the default color cycle
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Number of bars per group
    n_bars = len(data)

    # The width of a single bar
    bar_width = total_width / n_bars

    # List containing handles for the drawn bars, used for the legend
    bars = []

    # Iterate over all data
    for i, (name, values) in enumerate(data.items()):
        # The offset in x direction of that bar
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2

        # Draw a bar for every value of that type
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])#, tick_label=labels)

        # Add a handle to the last drawn bar, which we'll need for the legend
        bars.append(bar[0])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    # Draw legend if we need
    if legend:
        ax.legend(bars, data.keys())

def plot_fashion_mnist(features, labels, name):
    import umap
    fashionmnist_cats={0 :"T-shirt/top",1 :"Trouser",2 :"Pullover",3 :"Dress",4 :"Coat",5 :"Sandal",6 :"Shirt",7 :"Sneaker",8 :"Bag",9 :"Ankle boot"}
    
    mapper = umap.UMAP(metric='euclidean')
    embedding = mapper.fit_transform(features)

    label = pd.Series(labels)
    em = pd.DataFrame(embedding, columns=['A','B']) 
    em['label'] = pd.DataFrame(np.vectorize(fashionmnist_cats.get)(label))#label.astype(str)
    gg = ggplot(aes(x='A', y='B', color='label'), data=em)+geom_point()+ scale_color_discrete(guide=guide_legend())+ labs(x = "", y = "") 
    gg.save(name, dpi=300)
    #print(gg)
    return gg

NUMBER_OF_ALGORITHMS = 7
NUMBER_OF_MEASURES_REAL = 4
NUMBER_OF_MEASURES_SYNTH = 6
NUMBER_OF_RUNS = 10

def run_all(X, y, X_test, y_test, slow=True, percent_outliers=0, conf=False, verbose=True, return_results=False):
    resultats = np.zeros((NUMBER_OF_ALGORITHMS,NUMBER_OF_MEASURES_REAL)) #6 algos, 4 measures
        
    print("LDA")
    ldatest = LDA(method='distributional')
    st = time.time()
    ldatest.fit(X, pd.Series(y))
    resultats[0,0] = time.time()-st
    resultats[0,1:] = print_metrics(pd.Series(y_test), ldatest.predict(X_test, percent_outliers), conf=conf, verbose=verbose, ret=True)
    
    print("QDA")
    qdatest = QDA(method='distributional')
    st = time.time()
    qdatest.fit(X, pd.Series(y))
    resultats[1,0] = time.time()-st    
    resultats[1,1:] = print_metrics(pd.Series(y_test), qdatest.predict(X_test, percent_outliers), conf=conf, verbose=verbose, ret=True)
    
    print("RQDA-MCD")
    rqdamcdtest = RGQDA('MCD')
    st = time.time()
    #rqdamcdtest.fit(X, pd.Series(y), c=1)
    #resultats[2,0] = time.time()-st 
    #resultats[2,1:] = print_metrics(pd.Series(y_test), rqdamcdtest.predict(X_test, percent_outliers), conf=conf, verbose=verbose, ret=True)
    
    print("RGQDA-MCD")
    rgqdamcdtest = RGQDA('MCD')
    st = time.time()
    #rgqdamcdtest.fit(X, pd.Series(y))
    #resultats[3,0] = time.time()-st 
    #resultats[3,1:] = print_metrics(pd.Series(y_test), rgqdamcdtest.predict(X_test, percent_outliers), conf=conf, verbose=verbose, ret=True)   
    
    print("t-QDA")
    t_qdatest = t_QDA(method='distributional')
    st = time.time()
    t_qdatest.fit(X, pd.Series(y))
    resultats[4,0] = time.time()-st  
    resultats[4,1:] = print_metrics(pd.Series(y_test), t_qdatest.predict(X_test, percent_outliers), conf=conf, verbose=verbose, ret=True)

    print("FEMDA with p/trace")
    femdatest = FEMDA()
    femdatest.normalisation_method = 1
    st = time.time()
    femdatest.fit(X, pd.Series(y))
    resultats[5,0] = time.time()-st 
    resultats[5,1:] = print_metrics(pd.Series(y_test), femdatest.predict(X_test, percent_outliers), conf=conf, verbose=verbose, ret=True)
    
    print("FEMDA pre-normalised")
    femda_ntest = FEMDA_N()
    femda_ntest.normalisation_method = 1
    st = time.time()
    femda_ntest.fit(X, pd.Series(y))
    resultats[6,0] = time.time()-st 
    resultats[6,1:] = print_metrics(pd.Series(y_test), femda_ntest.predict(X_test, percent_outliers), conf=conf, verbose=verbose, ret=True)    
    
    models= {"LDA": ldatest, "QDA": qdatest, "RQDA-MCD": rqdamcdtest, "RGQDA-MCD": rgqdamcdtest, "t_QDA": t_qdatest, "FEMDA":femdatest, "FEMDA_N":femda_ntest}
    
    if return_results:
        return models, resultats
    else:
        return models

def test_all(models, X_test, y_test, percent_outliers=0):
    models = models if type(models) is not dict else list(models.values())
    
    for model in models:
        print(type(model).__name__)
        print_metrics(pd.Series(y_test), model.predict(X_test, percent_outliers=percent_outliers), conf=False)
    
"""
OLD MODELS:
print("t-LDA")
t_ldatest = t_LDA(method='distributional')
t_ldatest.fit(X, pd.Series(y))
print_metrics(pd.Series(y_test), t_ldatest.predict(X_test, percent_outliers), conf=conf, ret=True)

print("RGQDA-S")
rgqdastest = RGQDA('S-estimator')
rgqdastest.fit(X, pd.Series(y))
print_metrics(pd.Series(y_test), rgqdastest.predict(X_test, percent_outliers))
print("RQDA-M")
rqdatest = RGQDA('M-estimator')
rqdatest.fit(X, pd.Series(y), c=1)
print_metrics(pd.Series(y_test), rqdatest.predict(X_test, percent_outliers))

print("t-QDA-FEM")
t_qda_femtest = t_QDA_FEM(method='distributional')
st = time.time()
t_qda_femtest.fit(X, pd.Series(y))
resultats[3,0] = time.time()-st 
resultats[3,1:] = print_metrics(pd.Series(y_test), t_qda_femtest.predict(X_test, percent_outliers), conf=conf, verbose=verbose, ret=True)

print("QDA-FEM")
qda_femtest = QDA_FEM(method='distributional')
qda_femtest.fit(X, pd.Series(y))
print_metrics(pd.Series(y_test), qda_femtest.predict(X_test, percent_outliers), conf=conf)

models= {"LDA": ldatest, "QDA": qdatest, "RGQDA_M": rgqdatest, "RQDA_S": rqdastest, "t_QDA": t_qdatest, "FEMDA1":femda1test}#  #"RGQDA_M": rgqdatest, "RGQDA_S": rgqdastest, "RQDA_M": rqdatest, "RQDA_S": rqdastest,
"""