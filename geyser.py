import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import mixture
import time

# help from tutorial at:
# https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py

plotInitial = False
plotDistrib = True

data = pd.read_csv("data/data.txt", delimiter="\s+")

if plotDistrib:
    iterList = []
    timeList = []
    for i in range(0, 50):
        tic = time.clock()
        gmm = mixture.GaussianMixture(n_components=2, tol = 1e-4, init_params = 'kmeans', covariance_type='spherical').fit(data)
        toc = time.clock()
        timeList += [toc-tic]
        iterList += [gmm.n_iter_]

    print("Time = " + str(np.mean(timeList)))
    plt.hist(iterList, bins = "auto")
    plt.title("Distribution of model iterations with random ICs")
    plt.xlabel("Number of iterations")
    plt.ylabel("Frequency")
    plt.show()
else:
    tic = time.clock()
    gmm = mixture.GaussianMixture(n_components=2, tol = 1e-4, init_params = 'random', covariance_type='spherical').fit(data)
    toc = time.clock()
    print(toc-tic)

print(gmm.n_iter_)
print(gmm.means_)
if plotInitial:
    plt.plot(data.loc[:, 'eruptions'], data.loc[:, 'waiting'], 'b.')
    plt.plot(gmm.means_[0][0], gmm.means_[0][1], 'ro')
    plt.plot(gmm.means_[1][0], gmm.means_[1][1], 'ro')
    plt.title("Old Faithful Geyser Eruptions")
    plt.xlabel("Eruption length (min)")
    plt.ylabel("Time between eruptions")
    plt.show()