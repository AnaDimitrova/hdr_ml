import os
import nibabel as nib
import numpy as np

from sklearn.externals import joblib
import src.Constants as const

# Apply a given mask to the image data
def applyMask(imageData, mask):
	remainingData = imageData[:, :, :, 0][ mask ]
	return remainingData

# # Get the mask that will be used to remove the black background of the brain image
# def getMaskToRemoveBlackBackground(imageData):
# 	mask = np.where(imageData[:, :, :, 0]>0)
# 	return mask

# Calculate the squared error rate
def calculateSquaredError(reg, input, output):
	meanSquaredError = np.mean((reg.predict(input) - output) ** 2)
	return meanSquaredError

def usePrecomputedData(precomputedValuesPath, callFunction, *args):
	if not(os.path.exists(const.Precomputed_Directory)):
		os.makedirs(const.Precomputed_Directory)

	if  not(os.path.isfile(precomputedValuesPath)):
		result = callFunction(*args)
		joblib.dump(result, precomputedValuesPath)
	else:
		print("Using cached features...(to compute again delete '%s')" 
			%(precomputedValuesPath))
		result = joblib.load(precomputedValuesPath) 
	return result

# Transform input data that will be used for regression
def transformInputForRegression(data, numberOfCases, mask):
	#Initial input (only first image)
	input = applyMask(data[0].get_data(), mask)

	# Iterate through the files to apply mask and transform 
	# the 3 dimentianal data into 1 dimentianal array
	print ("Reading %d files, this may take a bit of time..." %(numberOfCases))
	for i in range(1,numberOfCases):

		print (".", end=" ", flush=True)
		processedImage = applyMask(data[i].get_data(), mask)

		#Every image becomes a row of the input matrix
		input = np.vstack([input, processedImage])

	return input

# Build mask from the first file of a data set.
def getMask(data):
	return np.where(data[0].get_data()[:, :, :, 0]>0);
