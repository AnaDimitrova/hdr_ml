import os
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import log_loss
from math import log
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

import Constants as const
import ProcessFiles as sp
import ReadData as rd
import ProcessData as pd

#######################################################
# 					Important!						  #		
# The following variables control what you want to be #
# processed and what not. The first time you run it,  #
# make sure that they are all set to TRUE. After that #
# you can disable them. If you want to preprocess the #
# actual features just set the property to true.	  #
# Note: Preprocessing the features is actually a slow #
# process so be mindful when you want to do that!	  #
#######################################################
Preprocess_Train_Files = False
Preprocess_Train_Features = False
Preprocess_Test_Files = False
Preprocess_Test_Features = False
#######################################################

# LEARN  
print("1. Starting reading train input data.")
sp.ProcessFiles(const.Train_Data_Path, const.Precomputed_Train_Directory, Preprocess_Train_Files);

print("2. Read or find features.")
bins = sp.BuildHistogramBins()
input = sp.ExtractFeaturesFromAllFiles(const.Precomputed_Train_Directory, const.Preprocessed_Train_Features_File, bins, Preprocess_Train_Features)
#Get training output
output = np.genfromtxt(const.Train_Target_File_Path, delimiter=',')

print("3. Reading test input data.")
sp.ProcessFiles(const.Test_Data_Path, const.Precomputed_Test_Directory, Preprocess_Test_Files)

print("4. Processing features for test input data.")
testInput = sp.ExtractFeaturesFromAllFiles(const.Precomputed_Test_Directory, const.Preprocessed_Test_Features_File, bins, Preprocess_Test_Features)


def ExecuteLearningHealth(input, output, testInput):
	features = SelectKBest(f_regression, k=const.Number_Of_Features_Project3_Health).fit(input, output);
	input = features.transform(input)
	reg = RidgeCV(normalize=True)
	reg.fit(input, output)

	predictionsOnTrainingData =  reg.predict(input) >= 0.5
	print("Health accuracy: {}".format(accuracy_score(predictionsOnTrainingData, output)) )

	testInputTransformed = features.transform(testInput)
	predictions = reg.predict(testInputTransformed)

	return predictions

def ExecuteLearningAge(input, output, testInput):
	features = SelectKBest(f_regression, k=const.Number_Of_Features_Project3_Age).fit(input, output);
	input = features.transform(input)
	reg = RidgeCV(normalize=True)
	reg.fit(input, output)

	predictionsOnTrainingData =  reg.predict(input) >= 0.5
	print("Age accuracy: {}".format(accuracy_score(predictionsOnTrainingData, output)) )

	testInputTransformed = features.transform(testInput)
	predictions = reg.predict(testInputTransformed)
	return predictions;

def ExecuteLearningGender(input, output, testInput):
	print("5.1. Starting reading train input data and mask...")
	# Read image files
	data = rd.GetAllFiles(const.Train_Data_Path)
	mask = pd.usePrecomputedData(const.Preprocessed_Mask_File,
								 lambda data: pd.getMask(data),
								 data)
	input = pd.usePrecomputedData(const.Preprocessed_Train_Input_File,
								  lambda data, trainSamples, mask: pd.transformInputForRegression(data, trainSamples,
																								  mask),
								  data, const.TRAIN_SAMPLES, mask)
	print("Finished reading train input data.")

	print("5.2. Doing PCA analysis...")
	pca = pd.usePrecomputedData(const.PCA_Analysis_File, lambda input: PCA(n_components=150).fit(input), input)
	print("Finished PCA analysis")

	print("5.3. Transforming input to selected features...")
	input = pca.transform(input)
	print("Finished transforming input")

	print("5.4. Fitting model...")
	features = SelectKBest(f_regression, k=const.Number_Of_Features_Project3_Gender).fit(input, output);

	input = features.transform(input)

	reg = RidgeCV(normalize=True)
	reg.fit(input, output)
	print("Model fitted")

	predictionsOnTrainingData =  reg.predict(input) >= 0.5
	print("Gender accuracy: {}".format(accuracy_score(predictionsOnTrainingData, output)) )

	print("5.5. Starting reading and transforming test input data...")
	testData = rd.GetAllFiles(const.Test_Data_Path)
	testInput = pd.usePrecomputedData(const.Preprocessed_Test_Input_File,
									  lambda data, testSamples, mask: pd.transformInputForRegression(data, testSamples,
																									 mask),
									  testData, const.TEST_SAMPLES, mask)
	testInput = pca.transform(testInput)
	testInputTransformed = features.transform(testInput)
	print("Finished reading and transforming train input data")

	predictions = reg.predict(testInputTransformed)
	return predictions


def createPrediction(result, value, middle):
	result = result >= middle
	return np.c_[np.arange(0,const.TEST_SAMPLES), [value] * const.TEST_SAMPLES, result]

genderOutput = output[:,0]; # male (0) / female (1)
ageOutput = output[:,1]; #  young (1) / old (0)
healthOutput = output[:,2]; # sick (0) / healthy (1)

#testInput = None
print("5. Learn Gender.")
genderResult = ExecuteLearningGender(input, genderOutput, testInput);
print("6. Learn Age.")
ageResult = ExecuteLearningAge(input, ageOutput, testInput);
print("7. Learn Health.")
healthResult = ExecuteLearningHealth(input, healthOutput, testInput);
print("8. Fix Predictions.")

#testPredValue = [False]

predictionsGender = createPrediction(genderResult, "gender",0.5);
#predictionsGender = np.c_[np.arange(0,const.TEST_SAMPLES), ["gender"] * const.TEST_SAMPLES, testPredValue * const.TEST_SAMPLES]
predictionsAge  = createPrediction(ageResult, "age", 0.5);
#predictionsAge = np.c_[np.arange(0,const.TEST_SAMPLES), ["age"] * const.TEST_SAMPLES, testPredValue * const.TEST_SAMPLES]
predictionsHealth = createPrediction(healthResult, "health", 0.5);
#predictionsHealth = np.c_[np.arange(0,const.TEST_SAMPLES), ["health"] * const.TEST_SAMPLES, testPredValue * const.TEST_SAMPLES]

predictions = np.empty((const.TEST_SAMPLES * 3, 4), dtype="<U21")
predictions[0::3, 1:4] = predictionsGender
predictions[1::3, 1:4] = predictionsAge
predictions[2::3, 1:4] = predictionsHealth

predictions[:,0] = np.arange(0, const.TEST_SAMPLES * 3)

np.savetxt(const.Test_Target_File_Path, predictions, delimiter=",", fmt="%s", header="ID,Sample,Label,Predicted", comments="")
print ("Result printed at '%s'." %(const.Test_Target_File_Path))



