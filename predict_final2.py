import os
import numpy as np
from sklearn.linear_model import LassoCV

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import log_loss
from math import log
from sklearn.pipeline import make_pipeline

import Constants as const
import ProcessFiles as sp

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
bins = sp.BuildHistogramBins();
input = sp.ExtractFeaturesFromAllFiles(const.Precomputed_Train_Directory, const.Preprocessed_Train_Features_File, bins, Preprocess_Train_Features)

#Get training output
output = np.genfromtxt(const.Train_Target_File_Path, delimiter='\n')

# Pick the best k features and adjust model
print("3. Selecting best features.")
features = SelectKBest(f_regression, k=const.Number_Of_Features_Project2).fit(input, output);

print("4. Transforming input to selected features.")
input = features.transform(input)

print("5. Fitting model.")
# Fit the linear mode with ridge regression including cross-validation with 20 folds
reg = LassoCV(normalize=True, max_iter=10000, cv=20, n_alphas=10000)
reg.fit(input, output)

# Check score
# Score doesn't really mean much at the moment.
print("Score: ", log_loss(output, reg.predict(input)))

# Read test input data
print("6. Starting reading test input data.")
sp.ProcessFiles(const.Test_Data_Path, const.Precomputed_Test_Directory, Preprocess_Test_Files)

print("7. Starting processing features for test input data.")
testInput = sp.ExtractFeaturesFromAllFiles(const.Precomputed_Test_Directory, const.Preprocessed_Test_Features_File, bins, Preprocess_Test_Features)

print("8. Transform test input to best features.")
testInputTransformed = features.transform(testInput)

print("9. Starting predicting test data.")
predictions = reg.predict(testInputTransformed)
predictions = np.maximum(predictions, const.RESULT_MIN_VALUE)
predictions = np.minimum(predictions, const.RESULT_MAX_VALUE)

predictions[predictions>0.8]=0.97
predictions[predictions<0.2]=0.07

#Format and save predictions as CSV
predictions = np.c_[np.arange(1,const.TEST_SAMPLES + 1), predictions]
np.savetxt(const.Test_Target_File_Path, predictions, delimiter=",", fmt="%i,%f", header="ID,Prediction", comments="")
print ("10. Result printed at '%s'." %(const.Test_Target_File_Path))



