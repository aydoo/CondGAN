import os
import numpy as np
import pandas as pd
from random import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def generate_data(num_classes = 2, num_features = 2, num_samples = 1000, seed = None):

    if seed is not None:
        np.random.seed(seed)

    mean_range = num_classes * 30
    std_range = num_classes * 2

    num_samples_per_classs = num_samples // num_classes

    X = None
    Y = None

    for i in range(num_classes):
        X_i = None

        for j in range(num_features):
            mean = np.random.randint(1, mean_range)
            std = np.random.randint(1, std_range)

            X_i_j = np.random.normal(mean, std, (num_samples_per_classs, 1))
            X_i = np.hstack((X_i, X_i_j)) if X_i is not None else X_i_j

        X = np.vstack((X, X_i)) if X is not None else X_i

        Y_i = np.full((num_samples_per_classs, 1), i+1)
        Y = np.vstack((Y, Y_i)) if Y is not None else Y_i

    return X, Y

def one_hot_encode(Y):
    return np.array(pd.get_dummies(np.squeeze(Y)))

def plot_data(X, Y):
    plt.figure()
    plt.scatter(X[:,0],X[:,1], label= list(np.squeeze(Y)), 
	                           c=list(np.squeeze(Y)), 
							   cmap = cm.get_cmap("viridis",len(np.unique(Y))),
							   alpha=0.1)
    plt.colorbar(ticks=np.unique(Y))
    plt.show()

if __name__ == '__main__':
	seed = 12

	num_classes = 5
	num_features = 2
	num_samples = 10000

	# Generate data
	X, Y = generate_data(num_classes, num_features, num_samples, seed)
	Y_encoded = one_hot_encode(Y)
	plot_data(X, Y)
