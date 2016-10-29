import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors.kde import KernelDensity
from sklearn.grid_search import GridSearchCV

import Constants as const

output = np.genfromtxt(const.Train_Target_File_Path, delimiter='\n').reshape(-1, 1)
myOutput = np.genfromtxt(const.Test_Target_File_Path, delimiter=',')[1:, 1].reshape(-1, 1)

params = {'bandwidth': np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(kernel='gaussian'), params)
grid.fit(output)

kde = grid.best_estimator_

samples = kde.sample(138)
samples = np.sort(samples, axis=0)
print(samples)
improvedOutput = np.zeros(138)
improvedOutput[myOutput.T.argsort()[0, :].T] = samples
improvedOutput = improvedOutput.reshape(-1, 1)

fig, ax = plt.subplots(1, 3)
ax[0].hist(output, 50, normed=1, facecolor='green', alpha=0.75)
ax[1].hist(myOutput, 50, normed=1, facecolor='red', alpha=0.75)
ax[2].hist(improvedOutput, 50, normed=1, facecolor='blue', alpha=0.75)
plt.subplots_adjust(left=0.15)
#plt.show()


improvedOutput = np.maximum(improvedOutput, 18)
improvedOutput = np.c_[np.arange(1,139), np.rint(improvedOutput)]
print(improvedOutput)
np.savetxt("./submissionExperimental.csv", improvedOutput, delimiter=",", fmt="%i", header="ID,Prediction", comments="")