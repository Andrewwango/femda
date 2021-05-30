import numpy as np
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from preprocess_utils import *
from postprocess_utils import *
from sklearn import metrics # AMII and ARI
import math, umap
from plotnine import *
from plotnine.data import *

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