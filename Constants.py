
import os

######################################################################	
# Important: Update the following variables to fit your environment. #
######################################################################

if(not "ML_PATH" in os.environ):
	raise Exception("Make sure you create a global variable \"ML_PATH\" for your project location in your ~/.bash_profile, or update Constants.py")

Enlistment_Location = os.environ['ML_PATH']
# You can specify the enlistment location here. Example below:
#Enlistment_Location = "/Users/jr/ETH/Machine Learning/"

Code_Location = Enlistment_Location + "/hdr_ml/"

Data_Location = Code_Location + "/data 3/"

Train_Data_Path = Data_Location + "/set_train/"
Test_Data_Path = Data_Location + "/set_test/"

Train_Target_File_Path = Data_Location + "targets.csv"
Test_Target_File_Path = Data_Location + "submission.csv"

######################################################################	
# 						END (important node)						 #
######################################################################


######################################################################
# 					Variables agnostic to custom setup 			 	 #
######################################################################

File_Extension = ".nii"

TRAIN_SAMPLES = 278
TEST_SAMPLES = 138

# Project 2 
# Make sure we don't have any 1s and 0s as results for the project.
RESULT_MIN_VALUE = 0.001;
RESULT_MAX_VALUE = 0.999;

# Project 1 
# RESULT_MIN_VALUE = 18;
# RESULT_MAX_VALUE = 100;

# Number of splits that will be discarded on each side of the Nii image
SPLIT_MARGIN = 30;
# Number of pizels to be cropped from the side of each slide
PIXELS_CROP_FROM_SIDE = 40;
# Number of segments to be created based on the X axis
SEGMENT_COUNT_X = 3
# Number of segments to be created based on the Y axis
SEGMENT_COUNT_Y = 3
# Number of bins to be use to create histograms
NUMBER_OF_BINS = 1000

# Number was assembled based on comparison 
# between squared error on the different feature number
Number_Of_Features_Project1 = 355000
Number_Of_Features_Project2 = 50
Number_Of_Features_Project3_Age = 40
Number_Of_Features_Project3_Health = 100
Number_Of_Features_Project3_Gender = 240000

Precomputed_Directory = Data_Location + '/PrecomputedData/'
Precomputed_Train_Directory = Precomputed_Directory + '/Train/'
Precomputed_Test_Directory = Precomputed_Directory + '/Test/'
Preprocessed_Train_Input_File = Precomputed_Directory + '/train_input.pkl'
Preprocessed_Test_Input_File = Precomputed_Directory + '/test_input.pkl'
Preprocessed_Mask_File = Precomputed_Directory + '/mask.pkl'
Preprocessed_Features_File = Precomputed_Directory + '/feature_selection.pkl'
Preprocessed_Train_Features_File = Precomputed_Directory + '/train_features.pkl'
Preprocessed_Test_Features_File = Precomputed_Directory + '/test_feature.pkl'
ç = Precomputed_Directory + '/feature_test_selection.pkl'
PCA_Analysis_File = Precomputed_Directory + '/pca_analysis.pkl'






