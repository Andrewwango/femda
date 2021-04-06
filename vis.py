import numpy as np
from matplotlib import pyplot as plt
from import_subset_datasets import *
from clustering_accuracy import acc
from sklearn import metrics # AMII and ARI

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

def print_metrics(true, pred, conf=False):
    dp = 5
    label = true.astype(int)
    print("Accuracy", round(acc(label, pred.astype(int)), dp), "ARI", round(metrics.adjusted_rand_score(label, pred.astype(str)), dp), "AMI", round(metrics.adjusted_mutual_info_score(label, pred.astype(str)), dp))
    if conf:
        print(metrics.confusion_matrix(true.astype(int), pred.astype(int)))
    
    
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

def plot_dataset(X, y, ax):
    for u in np.unique(y):
        ax.scatter(X[y==u,0],X[y==u,1])

def plot_models(X, y, X_test, models, lims=[[-10,10],[-10,10]]):
    many = type(models) is tuple and len(models)>1
    models = models if many else [models]
    cols = 5
    rows = 2*int((len(models))/cols+1)
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
        plot_dataset(X_test, model.predict(X_test), axs[one,i])
    plt.subplots_adjust(wspace=0, hspace=0)

def plot_means(models):
    for i,model in enumerate(models):
        means = model.means + (np.random.random_sample(size=model.means.shape)-0.5)*0.5
        plt.scatter(means[0,:], means[1,:], label=type(model).__name__)
    plt.legend()