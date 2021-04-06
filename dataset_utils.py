import numpy as np

def split(n, perc):
    a = int(np.floor(n*perc))
    b = int(np.ceil(n*(1-perc)))
    assert (a+b==n)
    return a,b

def contaminated(n, dist, contamination, loc, shape, df, hard):
    a,b = split(n, contamination)
    X1 = dist(loc=loc, shape=shape, df=df).rvs(size=b)
    if contamination == 0:
        return X1
    else:
        X1c = dist(loc=(-9 if hard else 9)*loc, shape=4*shape, df=df).rvs(size=a)
        return np.vstack([X1, X1c])

def contaminate_dataset(a, perc):
    idx = np.random.choice(len(a),int(np.floor(len(a)*perc)),replace=False)
    a[idx, :] = 10*a[idx, :]
    return a

def mislabelled(n, mislabelling, labels):
    a,b = split(n, mislabelling)
    return np.hstack([np.random.permutation(np.hstack([i*np.ones((b)), labels[np.random.randint(0, len(labels), (a))]])) for i in labels])

def flip_bits(a, perc):
    idx = np.random.choice(len(a),int(np.floor(len(a)*perc)),replace=False)
    a[idx] = 1-a[idx]
    return a