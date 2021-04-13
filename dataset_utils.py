import numpy as np
from scipy import linalg
def split(n, perc):
    a = int(np.floor(n*perc))
    #b = int(np.ceil(n*(1-perc)))
    b = n-a
    #print(a,b,n)
    assert (a+b==n)
    return a,b

def apply_outlierness(outliernesses, data):
    outlierness = np.hstack(outliernesses)
    print(outlierness.shape, data.shape)
    assert(len(outlierness)==data.shape[0])
    data[outlierness] = -1
    return data

def contaminated(n, dist, contamination, loc, shape, df, hard):
    a,b = split(n, contamination)
    X1 = dist(loc=loc, shape=shape, df=df).rvs(size=b)
    if contamination == 0:
        return X1, np.zeros((X1.shape[0]))>1
    else:
        X1c = dist(loc=(-9 if hard else 9)*loc, shape=4*shape, df=df).rvs(size=a)
        return np.vstack([X1, X1c]), np.hstack([np.zeros((b))>1, np.ones((a))>0])

def combine_dataset(X1, X2, X1perc):
    assert(X1.shape[0] == X2.shape[0])
    a,b = split(X1.shape[0], X1perc)
    out = np.hstack([X1[:a], X2[a:]]) if X1.ndim == 1 else np.vstack([X1[:a,:], X2[a:,:]])
    #np.random.shuffle(out)
    return out
    
def contaminate_dataset(a, perc):
    idx = np.random.choice(len(a),int(np.floor(len(a)*perc)),replace=False)
    a[idx, :] = 10*a[idx, :]
    outlierness = np.zeros((a.shape[0]))
    outlierness[idx] = 1
    return a, outlierness>0

def mislabelled(n, mislabelling, labels):
    a,b = split(n, mislabelling)
    return np.hstack([np.random.permutation(np.hstack([i*np.ones((b)), labels[np.random.randint(0, len(labels), (a))]])) for i in labels])

def flip_bits(a, perc):
    idx = np.random.choice(len(a),int(np.floor(len(a)*perc)),replace=False)
    a[idx] = 1-a[idx]
    return a

def toeplitz(r, p):
    return linalg.toeplitz(np.array([r**j for j in range(p)]))