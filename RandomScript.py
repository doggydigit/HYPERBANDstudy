#import numpy as np
#import matplotlib.pyplot as plt
import os
import sys
#import math
#from sklearn import svm
from myfunctions import *
from mnist import MNIST
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('../scripts'))

mndata = MNIST('../../Data/MNIST')
mnist_data, mnist_labels = mndata.load_training()
nr_data = len(mnist_data)
nr_features = len(mnist_data[1])
nr_traindata = int(0.7*nr_data)
nr_validationdata = nr_data - nr_traindata
traindata = mnist_data[:nr_traindata]
trainlabels = mnist_labels[:nr_traindata]
validationdata = mnist_data[nr_validationdata:]
validationlabels = mnist_labels[nr_validationdata:]
print "Data loaded"
hyperparameters = {"continuous": [("coef0", [-100.0, 100.0])],
                   "exponential": [("C", [0.0001, 1000]),
                                 ("gamma", [0.001, 10.0])],
                   "discrete": [("kernel", ["poly", "sigmoid", "rbf"])],
                   "integer": [("degree", [1, 5])]}
nr_parameters = 5
# Random search
total_nr_configurations = 50
configurations = [None]*total_nr_configurations
for i in range(0, total_nr_configurations):
    configurations[i] = get_randomconfiguration(hyperparameters)

print("Nr of tested configurations is {}".format(total_nr_configurations))
nrtrain = 6000
nrval = 1000
print "Starting validation of configurations"
errors = validate_configurations(configurations, traindata[:nrtrain], trainlabels[:nrtrain],
                                 validationdata[:nrval], validationlabels[:nrval])

besterror = 1
for i in range(0,total_nr_configurations):
    if errors[i] < besterror:
        besterror = errors[i]
        bestconfiguration = configurations[i]

print "Best configuration is:"
print bestconfiguration
print("With best error:{}".format(besterror))