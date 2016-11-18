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
import SplitData as sp


#######################################################
# 					Important!						  #		
# The following variables control what you want to be #
# processed and what not. The first time you run it,  #
# make sure that they are all set to true. After that #
# you can disable them. If you want to preprocess the #
# actual features just set the property to true.	  #
# Note: Preprocessing the features is actually a slow #
# process so be mindful when you want to do that!	  #
#######################################################
Preprocess_Train_Files = False;
Preprocess_Train_Features = False;
Preprocess_Test_Files = False;
Preprocess_Test_Features = False;
#######################################################

# LEARN  
print("1. Starting reading train input data...")
sp.ProcessFiles(const.Train_Data_Path, const.Precomputed_Train_Directory, Preprocess_Train_Files);
print("Finished reading train input data.")


print("2. Read or find features...")
bins = sp.BuildHistogramBins();
input = sp.ExtractFeaturesFromAllFiles(const.Precomputed_Train_Directory, const.Preprocessed_Train_Features_File, bins, Preprocess_Train_Features)

#Get training output
output = np.genfromtxt(const.Train_Target_File_Path, delimiter='\n')

# Pick the best k features and adjust model
print("3. Selecting best features...")
features = SelectKBest(f_regression, k=const.Number_Of_Features_New).fit(input, output);
print("Best features selected.")

print("4. Transforming input to selected features...")
input = features.transform(input)
print("Finished transforming input")

print("5. Fitting model...")
# Fit the linear mode with ridge regression including Leave-One-Out cross-validation
reg = LassoCV(normalize=True)
# weights = output.copy()
# weights[weights==0]=10
reg.fit(input, output)

print("Model fitted")

# Check score
# Score doesn't really mean much at the moment.
# print("Score: ", reg.score(input, reg.predict(input)))

# Read test input data
print("6. Starting reading test input data...")
# TODO: fix the check that actually the files exist
sp.ProcessFiles(const.Test_Data_Path, const.Precomputed_Test_Directory, Preprocess_Test_Files)
print("Finished reading test input data.")

print("7. Starting processing features for test input data...")
testInput = sp.ExtractFeaturesFromAllFiles(const.Precomputed_Test_Directory, const.Preprocessed_Test_Features_File, bins, Preprocess_Test_Features)
print("Finished reading train input data")

print("8. Transform test input to best features...")
testInputTransformed = features.transform(testInput)
print("Finished transforming test input...")

print("9. Starting predicting test data...")
predictions = reg.predict(testInputTransformed)
predictions = np.maximum(predictions, const.RESULT_MIN_VALUE)
predictions = np.minimum(predictions, const.RESULT_MAX_VALUE)
print("Finished predicting test data...")

#Format and save predictions as CSV
predictions = np.c_[np.arange(1,const.TEST_SAMPLES + 1), predictions]
np.savetxt(const.Test_Target_File_Path, predictions, delimiter=",", fmt="%i,%f", header="ID,Prediction", comments="")
print ("10. Result printed at '%s'." %(const.Test_Target_File_Path))












