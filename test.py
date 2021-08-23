from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.estimator_checks import check_estimator
import numpy as np
from femda import FEMDA
f = FEMDA()
print(check_estimator(f))