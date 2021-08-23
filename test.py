from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator
import numpy as np
#from femda import FEMDA
#f = FEMDA()
#print(check_estimator(f))
class test:
    def hi(self, a,b,c):
        print(a+b+c)
    
    def hi(self, a,b):
        print(a+b)

t = test()
t.hi(6,7,8)
t.hi(6,7)