import nibabel
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

import ReadData as rd
import Constants as const

def calculateFeatureForX(imageData, threshold):

	#Take a slice (90 <= z <= 100)
	aSlice = imageData[:, :, 90:100, 0]

	#Remove background
	noBlack = aSlice[ np.where(aSlice>0) ]

	#Count every with intensity higher than threshold
	return ((noBlack > threshold).sum())

def generateInputFor(path, numberOfCases, threshold):
	input = np.empty(shape=(numberOfCases))

	#Read image files
	data = rd.GetAllFiles(path)

	for i in range(0,numberOfCases):

		#Count every with intensity higher than threshold
		input[i] = calculateFeatureForX(data[i].get_data(), threshold)

	#Reshape
	return input.reshape((numberOfCases, 1))


def plotRegressionVsActualData():
	plt.scatter(input, output,  color='black')

	x_plot = np.linspace(np.amin(input), np.amax(input), 100)[:, np.newaxis]
	plt.plot(x_plot, model.predict(x_plot), color='blue',
         linewidth=3)

	plt.xticks(())
	plt.yticks(())
	plt.show()


#Initial bogus values
minMeanSquaredError = 999
bestThreshold = 0

for threshold in range (300, 701, 10):

	input = generateInputFor(const.Train_Data_Path, const.TRAIN_SAMPLES, threshold)

	#Get training output
	output = np.genfromtxt(const.Train_Target_File_Path, delimiter='\n')

	#Quadratic polynomial fit
	model = make_pipeline(PolynomialFeatures(2), Ridge())
	model.fit(input, output)

	#Linear Regression
	#model = linear_model.LinearRegression()
	#model.fit(input, output)

	meanSquaredError = np.mean((model.predict(input) - output) ** 2)
	print("Mean squared error: %.2f"
	      % meanSquaredError )

	if(meanSquaredError < minMeanSquaredError):
		bestThreshold = threshold

plotRegressionVsActualData()

#PREDICT

#Generate test input
testInput = generateInputFor(const.Test_Data_Path, const.TEST_SAMPLES, bestThreshold)

#Run against model
predictions = np.rint(model.predict(testInput))

#Over 18 age limit
predictions = np.maximum(predictions, 18)

#Format and save predictions as CSV
predictions = np.c_[np.arange(1,139), predictions]
print(predictions)
np.savetxt(const.Test_Target_File_Path, predictions, delimiter=",", fmt="%i", header="ID,Prediction", comments="")
