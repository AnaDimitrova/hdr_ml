import nibabel
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import BaggingRegressor

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

#Compute input array and a mask to eliminate background (probable unnecessary given the statistical tests)
result = generateInputFor(const.Train_Data_Path, const.TRAIN_SAMPLES, None)
input=result['input']
mask=result['mask']

#Get training output
output = np.genfromtxt(const.Train_Target_File_Path, delimiter='\n')

#Pick only the best k features and adjust model
ps = SelectKBest(f_regression, k=const.Number_Of_Features).fit(input, output)
input = ps.transform(input)

reg = linear_model.RidgeCV(normalize=True)
#reg = BaggingRegressor(linear_model.RidgeCV(normalize=True), max_features=0.04)
reg.fit(input, output)

#Print mean squared error
#	meanSquaredError = np.mean((model.predict(input) - output) ** 2)
#	print("Mean squared error: %.2f"
#	      % meanSquaredError )


#PREDICT

##Generate test input
testInput = generateInputFor(const.Test_Data_Path, const.TEST_SAMPLES, mask)['input']
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
