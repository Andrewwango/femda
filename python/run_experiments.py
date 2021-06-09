import numpy as np
import pandas as pd
import random, os, time, csv, warnings, math
from sklearn import metrics, decomposition, discriminant_analysis
from scipy import stats, special, optimize
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore') # AMI warning

import matplotlib
import matplotlib.pyplot as plt

from preprocess_utils import *
from postprocess_utils import *
from vis import *
from literature_models import *
from fit import *
from femda import *

NUMBER_OF_ALGORITHMS = 7
NUMBER_OF_MEASURES_REAL = 4
NUMBER_OF_MEASURES_SYNTH = 6
NUMBER_OF_RUNS = 10

def run_algorithms(X, y, X_test, y_test, slow=True, percent_outliers=0, conf=False, verbose=True, return_results=False):
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
    rqdamcdtest.fit(X, pd.Series(y), c=1)
    resultats[2,0] = time.time()-st 
    resultats[2,1:] = print_metrics(pd.Series(y_test), rqdamcdtest.predict(X_test, percent_outliers), conf=conf, verbose=verbose, ret=True)
    
    print("RGQDA-MCD")
    rgqdamcdtest = RGQDA('MCD')
    st = time.time()
    rgqdamcdtest.fit(X, pd.Series(y))
    resultats[3,0] = time.time()-st 
    resultats[3,1:] = print_metrics(pd.Series(y_test), rgqdamcdtest.predict(X_test, percent_outliers), conf=conf, verbose=verbose, ret=True)   
    
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

def test_models(models, X_test, y_test, percent_outliers=0):
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