import os
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV

from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
from math import log
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import ReadData as rd
import Constants as const
import ProcessData as pd

# LEARN  
print("1. Starting reading train input data and mask...")
# Read image files
data = rd.GetAllFiles(const.Train_Data_Path)
mask = pd.usePrecomputedData(const.Preprocessed_Mask_File,
	lambda data: pd.getMask(data),
	data)
input = pd.usePrecomputedData(const.Preprocessed_Train_Input_File,
	lambda data, trainSamples, mask: pd.transformInputForRegression(data, trainSamples, mask),
	data, const.TRAIN_SAMPLES, mask)
print("Finished reading train input data.")

#Get training output
output = np.genfromtxt(const.Train_Target_File_Path, delimiter='\n')

#Pick the best k features and adjust model
#print("2. Selecting best features...")
#ps = pd.usePrecomputedData(const.Preprocessed_Features_File,
#	lambda input, output:
#	SelectKBest(f_regression, k=const.Number_Of_Features_Project1).fit(input, output),
#	input, output)#
#print("Best features selected.")

print("2. Doing PCA analysis...")
pca = pd.usePrecomputedData(const.PCA_Analysis_File, lambda input: PCA(n_components=150).fit(input), input)
print("Finished PCA analysis")

# Transforming input
print("3. Transforming input to selected features...")
input = pca.transform(input)
print("Finished transforming input")

print("4. Fitting model...")
# Fit the linear mode with ridge regression including Leave-One-Out cross-validation
reg = RidgeCV(normalize=True)
weights = output.copy()
weights[weights==0]=10
reg.fit(input, output, weights)
print("Model fitted")

#Check score
print("Score: ", reg.score(input, reg.predict(input)))

# Read test input data
print("5. Starting reading test input data...")
testData = rd.GetAllFiles(const.Test_Data_Path)
testInput = pd.usePrecomputedData(const.Preprocessed_Test_Input_File,
	lambda data, testSamples, mask: pd.transformInputForRegression(data, testSamples, mask),
	testData, const.TEST_SAMPLES, mask)
#testInputTransformed = ps.transform(testInput)
testInputTransformed = pca.transform(testInput)
print("Finished reading train input data")

print("6. Starting predicting test data...")
predictions = reg.predict(testInputTransformed)
predictions = np.maximum(predictions, const.RESULT_MIN_VALUE)
predictions = np.minimum(predictions, const.RESULT_MAX_VALUE)
print("Finished predicting test data...")

#Format and save predictions as CSV
predictions = np.c_[np.arange(1,const.TEST_SAMPLES + 1), predictions]
np.savetxt(const.Test_Target_File_Path, predictions, delimiter=",", fmt="%i,%f", header="ID,Prediction", comments="")
print ("7. Result printed at '%s'." %(const.Test_Target_File_Path))



