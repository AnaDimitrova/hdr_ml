import os
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

import ReadData as rd
import Constants as const
import ProcessData as pd

# LEARN  
print("1. Starting reading train input data and mask...")
# Read image files
data = rd.GetAllFiles(const.Train_Data_Path)
mask = pd.usePrecomputedData(const.Preprocessed_Mask_File,
	lambda data: pd.getMask(data),
	data);
input = pd.usePrecomputedData(const.Preprocessed_Train_Input_File,
	lambda data, trainSamples, mask: transformInputForRegression(data, trainSamples, mask),
	data, const.TRAIN_SAMPLES, mask);
print("Finished reading train input data.")

#Get training output
output = np.genfromtxt(const.Train_Target_File_Path, delimiter='\n')

#Pick the best k features and adjust model
print("2. Selecting best features...")
ps = pd.usePrecomputedData(const.Preprocessed_Features_File,
	lambda input, output: 
	pd.SelectKBest(f_regression, k=const.Number_Of_Features).fit(input, output),
	input, output)
print("Best features selected.")

# Transforming input
print("3. Transforming input to selected features...")
input = ps.transform(input)
print("Finished transforming input")

print("4. Fitting model...")
# Fit the linear mode with ridge regression including Leave-One-Out cross-validation
reg = RidgeCV(normalize=True)
reg.fit(input, output)
print("Model fitted")

# Read test input data
print("5. Starting reading test input data...")
testData = rd.GetAllFiles(const.Test_Data_Path)
testInput = pd.usePrecomputedData(const.Preprocessed_Test_Input_File,
	lambda data, testSamples, mask: pd.transformInputForRegression(data, testSamples, mask),
	testData, const.TEST_SAMPLES, mask)
testInputTransformed = ps.transform(testInput)
print("Finished reading train input data")

print("6. Starting predicting test data...")
#Predict and round to integer
predictions = reg.predict(testInputTransformed)
#Apply over 18 age limit
predictions = np.maximum(predictions, 18)
print("Finished predicting test data...")

#Format and save predictions as CSV
predictions = np.c_[np.arange(1,const.TEST_SAMPLES + 1), predictions]
np.savetxt(const.Test_Target_File_Path, predictions, delimiter=",", fmt="%i,%f", header="ID,Prediction", comments="")
#print (predictions)
print ("7. Result printed at '%s'." %(const.Test_Target_File_Path))



