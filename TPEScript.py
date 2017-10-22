import time
from hpsklearn import HyperoptEstimator, svc
from sklearn import svm

if use_hpsklearn:
    estim = HyperoptEstimator(classifier=svc('mySVC'))
else:
    estim = svm.SVC()