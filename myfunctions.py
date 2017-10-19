import numpy as np
import math
from sklearn import svm

def get_configuration(hyperparameters):
	configuration = ()
	for x in range(0,len(hyperparameters["continuous"])):
		configuration = configuration + (np.random.uniform(hyperparameters["continuous"][x][1][0],hyperparameters["continuous"][x][1][1]),)	
	for x in range(0,len(hyperparameters["exponential"])):
		configuration = configuration + (10**(np.random.uniform(math.log(hyperparameters["exponential"][x][1][0],10),math.log(hyperparameters["exponential"][x][1][1],10))),)
	for x in range(0,len(hyperparameters["discrete"])):
		configuration = configuration + (np.random.choice(hyperparameters["discrete"][x][1]),)
	for x in range(0,len(hyperparameters["integer"])):
		configuration = configuration + (np.random.randint(hyperparameters["integer"][x][1][0],hyperparameters["integer"][x][1][1]),)
	return configuration

def validate_configurations(configurations,traindata,trainlabels,validationdata,validationlabels):
	errors = [0]*len(configurations)
	for i in range(0,len(configurations)):
		clf = svm.SVC(gamma=configurations[i][2], C=configurations[i][1],kernel=str(configurations[i][3]),coef0=float(configurations[i][0]), degree=float(configurations[i][4]))
		clf.fit(traindata,trainlabels)
		prediction = clf.predict(validationdata)
		for j in range(0,len(validationdata)):
			if round(prediction[j]) != validationlabels[j]:
				errors[i] += 1
	return [error / len(validationlabels) for error in errors]

def keep_best_configurations(configurations, errors,eta):
	nr_best = int(len(configurations))
	best_configurations = configurations[:nr_best]
	best_errors = errors[:nr_best]
	worst_best = best_errors.index(max(best_errors))
	for i in range(nr_best,len(configurations)):
		if errors[i] < best_errors[worst_best]:
			best_errors[worst_best] = errors[i]
			best_configurations[worst_best] = configurations[i]
	return best_configurations
