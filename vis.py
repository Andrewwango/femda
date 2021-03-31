import numpy as np
from matplotlib import pyplot as plt

def lda_grid(em, mapper):
    AA, BB = np.meshgrid(np.linspace(em.A.min(), em.A.max(), 50),
                     np.linspace(em.B.min(), em.B.max(), 50))
    return AA, BB, mapper.inverse_transform(np.c_[AA.ravel(), BB.ravel()])

def plot_contours_UMAP(gg, lda, AA, BB, grid):
    print(grid.shape)
    cmaps = ['Reds','Greens','Blues','Purples','Oranges']
    cnt = lda.predict_proba(grid)
    fig = gg.draw()
    ax = fig.get_axes()[0]
    for i in range(cnt.shape[1]):
        ax.contour(AA, BB, cnt[:,i].reshape(AA.shape), cmap=cmaps[i])#, [0.5])
        
def plot_regions_UMAP(gg, lda, AA, BB, grid):
    colors = ['r','g','b','c','y']
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

def print_metrics(true, pred):
    print("Accuracy", acc(true.astype(int), pred.astype(int)))
    print(metrics.confusion_matrix(true.astype(int), pred.astype(int)))
    
def plot_contours(X, f, ax):
    AA, BB = np.meshgrid(np.linspace(X[:,0].min(), X[:,0].max(), 50),
                 np.linspace(X[:,1].min(), X[:,1].max(), 50))
    cmaps = ['Reds','Greens','Blues','Purples','Oranges']
    grid = np.c_[AA.ravel(), BB.ravel()]
    cnt = f(grid)
    for i in range(cnt.shape[1]):
        ax.contour(AA, BB, cnt[:,i].reshape(AA.shape), cmap=cmaps[i])#, [0.5])
        ax.contour(AA, BB, cnt[:,i].reshape(AA.shape), levels=[0.5], linewidths=3)#, [0.5])