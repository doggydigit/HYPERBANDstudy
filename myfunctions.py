import numpy as np
import math
from sklearn import svm


def get_randomconfiguration(hyperparameters):
    configuration = ()
    for x in range(0, len(hyperparameters["continuous"])):
        configuration += (np.random.uniform(hyperparameters["continuous"][x][1][0],
                                            hyperparameters["continuous"][x][1][1]),)

    for x in range(0, len(hyperparameters["exponential"])):
        configuration += (10**(np.random.uniform(math.log(hyperparameters["exponential"][x][1][0], 10),
                                                 math.log(hyperparameters["exponential"][x][1][1], 10))),)

    for x in range(0, len(hyperparameters["discrete"])):
        configuration += (np.random.choice(hyperparameters["discrete"][x][1]),)

    for x in range(0, len(hyperparameters["integer"])):
        configuration += (np.random.randint(hyperparameters["integer"][x][1][0],
                                            hyperparameters["integer"][x][1][1]),)
    return configuration


def validate_configurations(configurations, traindata, trainlabels, validationdata, validationlabels):
    errors = [0]*len(configurations)
    for i in range(0, len(configurations)):
        print configurations[i]
        clf = svm.SVC(gamma=configurations[i][2],
                      C=configurations[i][1],
                      kernel=str(configurations[i][3]),
                      coef0=float(configurations[i][0]),
                      degree=float(configurations[i][4]))
        clf.fit(traindata, trainlabels)
        prediction = clf.predict(validationdata)
        for j in range(0, len(validationdata)):
            if round(prediction[j]) != validationlabels[j]:
                errors[i] += 1
        print("Configuration {} produced {} errors from {} validation sets".format(i, errors[i], len(validationlabels)))
    return [float(error) / len(validationlabels) for error in errors]


def keep_best_configurations(configurations, errors, eta):
    nr_best = int(len(configurations)/eta)
    best_configurations = configurations[:nr_best]
    best_errors = errors[:nr_best]
    worst_best = best_errors.index(max(best_errors))
    for i in range(nr_best, len(configurations)):
        if errors[i] < best_errors[worst_best]:
            best_errors[worst_best] = errors[i]
            best_configurations[worst_best] = configurations[i]
    return best_configurations


def get_paramvector(a, b, n, ptype):
    if ptype == "exponential":
        a = math.log(a, 10)
        b = math.log(b, 10)
    cnst = (b-a)/(n+1)
    p = [None] * n
    for i in range(0, n):
        p[i] = cnst*(i+1)+a

    if ptype == "exponential":
        for i in range(0, n):
            p[i] = 10**p[i]
    return p


def get_parametergrid(hyperparameters, n):
    coef0 = get_paramvector(hyperparameters["continuous"][0][1][0],
                            hyperparameters["continuous"][0][1][1],
                            n, "continuous")
    c = get_paramvector(hyperparameters["exponential"][0][1][0],
                        hyperparameters["exponential"][0][1][1],
                        n, "exponential")
    gamma = get_paramvector(hyperparameters["exponential"][1][1][0],
                            hyperparameters["exponential"][1][1][1],
                            n, "exponential")
    degree = range(1, n+1, 1)
    return coef0, c, gamma, degree
