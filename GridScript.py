import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
from sklearn import svm
module_path = os.path.abspath(os.path.join('../scripts'))
if module_path not in sys.path:
    sys.path.append(module_path)
from myfunctions import *
from mnist import MNIST
mndata = MNIST('../../Data/MNIST')
mnist_data, mnist_labels = mndata.load_training()
nr_data = len(mnist_data)
nr_features = len(mnist_data[1])
nr_traindata = int(0.5*nr_data)
nr_validationdata = nr_data - nr_traindata
traindata = mnist_data[:nr_traindata]
trainlabels = mnist_labels[:nr_traindata]
validationdata = mnist_data[nr_validationdata:]
validationlabels = mnist_labels[nr_validationdata:]
print "Data loaded"
hyperparameters = {"continuous":[("coef0",[-100,100])],
                   "exponential":[("C",[0.0001,1000]),
                                 ("gamma",[0.001,10])],
                   "discrete":[("kernel",["poly","sigmoid","rbf"])],
                   "integer":[("degree",[1,5])]}
nr_parameters = 5
# Grid search
N = 4
coef0, C, gamma, degree = get_parametergrid(hyperparameters, N)
print coef0,C,gamma,degree
nr_kernelparam = [4, 2, 3]

total_nr_configurations = 0
for k in range(0, len(nr_kernelparam)):
    total_nr_configurations += N ** nr_kernelparam[k]
print("Nr of tested configurations is {}".format(total_nr_configurations))
configurations = [None] * total_nr_configurations

c = 0
l = N ** 2
for k in range(0, len(nr_kernelparam)):
    for j in range(0, (N ** nr_kernelparam[k]) / l):
        par = (coef0[j % N],)
        for i in range(0, l):
            configurations[c] = par
            c += 1

for c in range(0, total_nr_configurations):
    configurations[c] += (C[c % N],)

c = 0
l = N
for k in range(0, len(nr_kernelparam)):
    for j in range(0, (N ** nr_kernelparam[k]) / l):
        par = (gamma[j % N],)
        for i in range(0, l):
            configurations[c] += par
            c += 1

c = 0
for k in range(0, len(nr_kernelparam)):
    for i in range(0, N ** nr_kernelparam[k]):
        configurations[c] += (hyperparameters["discrete"][0][1][k],)
        c += 1

c = 0
l = N ** 3
for k in range(0, len(nr_kernelparam)):
    if nr_kernelparam[k] >= 4:
        for j in range(0, N):
            par = (degree[j],)
            for i in range(0, l):
                configurations[c] += par
                c += 1
    else:
        for j in range(0, N ** nr_kernelparam[k]):
            configurations[c] += (1,)
            c += 1

nrtrain = 600
nrval = 300
print "Starting validation of configurations"
errors = validate_configurations(configurations, traindata[:nrtrain], trainlabels[:nrtrain],
                                 validationdata[:nrval], validationlabels[:nrval])

besterror = 1
for i in range(0, total_nr_configurations):
    if errors[i] < besterror:
        besterror = errors[i]
        bestconfiguration = configurations[i]

print "Best configuration is:"
print bestconfiguration
print("With best error:{}".format(besterror))