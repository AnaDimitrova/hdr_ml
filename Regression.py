import nibabel
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
import os
from sklearn.model_selection import KFold

import ReadData as rd
import Constants as const

def applyMask(imageData, mask):

	#Remove background (perhaps unnecessary)
	noBlack = imageData[:, :, :, 0][ mask ]

	return noBlack

def generateInputFor(path, numberOfCases, mask):
	
	#Read image files
	data = rd.GetAllFiles(path)

	#Calculate mask to remove background from first input
	if(mask == None):
		mask = np.where(data[0].get_data()[:, :, :, 0]>0)

	#Initial input (only first image)
	input = applyMask(data[0].get_data(), mask)

	for i in range(1,numberOfCases):

		print(i)
		processedImage = applyMask(data[i].get_data(), mask)

		#Every image becomes a row of the input matrix
		input = np.vstack([input, processedImage])

	

	return {'input': input, 'mask': mask}


#LEARN
print("Starting train input data read...")
if  not(os.path.isfile('input.pkl')):
	#Compute input array and a mask to eliminate background (probable unnecessary given the statistical tests)
	result = generateInputFor(const.Train_Data_Path, const.TRAIN_SAMPLES, None)
	input=result['input']
	mask=result['mask']
	joblib.dump(input, 'input.pkl')
	joblib.dump(mask, 'mask.pkl')
else:
	print("Using cached train input data...")
	input = joblib.load('input.pkl') 
	mask = joblib.load('mask.pkl') 
print("Finished reading train input data")

#Get training output
output = np.genfromtxt(const.Train_Target_File_Path, delimiter='\n')

#Pick only the best k features and adjust model
print("Selecting best features...")
if  not(os.path.isfile('feature_selection.pkl')):
	ps = SelectKBest(f_regression, k=const.Number_Of_Features).fit(input, output)
	joblib.dump(ps, 'feature_selection.pkl')
else:
	print("Using cached features...")
	ps = joblib.load('feature_selection.pkl') 
print("Best features selected")

input = ps.transform(input)

print("Fitting model...")
reg = linear_model.RidgeCV(normalize=True, cv=KFold(n_splits=10))
#reg = linear_model.BayesianRidge(normalize=True)
#reg = svm.SVR(epsilon=0.5, kernel='linear', C=100)

reg.fit(input, output)

print("Model fitted")

#Print mean squared error
meanSquaredError = np.mean((reg.predict(input) - output) ** 2)
print("Mean squared error: %.2f"
      % meanSquaredError )


#PREDICT
print("Starting test input data read...")
if  not(os.path.isfile('testInput.pkl')):
	##Generate test input
	testInput = generateInputFor(const.Test_Data_Path, const.TEST_SAMPLES, mask)['input']
	joblib.dump(testInput, 'testInput.pkl')
else:
	print("Using cached test input data...")
	testInput = joblib.load('testInput.pkl') 
print("Finished reading train input data")

testInputTransformed = ps.transform(testInput)

#Predict and round to integer
predictions = np.rint(reg.predict(testInputTransformed))

#Apply over 18 age limit
predictions = np.maximum(predictions, 18)

#OUTPUT

#Format and save predictions as CSV
predictions = np.c_[np.arange(1,139), predictions]
print(predictions)
np.savetxt(const.Test_Target_File_Path, predictions, delimiter=",", fmt="%i", header="ID,Prediction", comments="")
