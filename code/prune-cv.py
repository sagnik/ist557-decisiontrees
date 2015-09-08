"""
============================================
Cross validated scores of the author dataset
============================================

"""
print __doc__

import numpy as np
from sklearn.datasets import load_boston
from sklearn import tree
import pickle


def plot_pruned_path(scores, with_std=False):
    """Plots the cross validated scores versus the number of leaves of trees"""
    import matplotlib.pyplot as plt
    means = np.array([np.mean(s) for s in scores])
    stds = np.array([np.std(s) for s in scores]) / np.sqrt(len(scores[1]))

    x = range(len(scores) + 1, 1, -1)

    plt.plot(x, means)
    if with_std:
        plt.plot(x, means + 2 * stds, lw=1, c='0.7')
        plt.plot(x, means - 2 * stds, lw=1, c='0.7')

    plt.xlabel('Number of leaves')
    plt.ylabel('Cross validated score')

    plt.show()


print "loading training dataset features"
traindata=np.asarray(pickle.load(open("data/traindata-allfeatures.list","r"))).astype(np.float)
print "loading class labels for training dataset"
target=np.asarray(pickle.load(open("data/target.list","r"))).astype(np.float)

clf = tree.DecisionTreeClassifier()

#Compute the cross validated scores
scores = tree.prune_path(clf, traindata, target,
                                    max_n_leaves=10, n_iterations=5,
                                    random_state=0)

plot_pruned_path(scores)
