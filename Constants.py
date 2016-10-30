
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

Train_Data_Path = Code_Location + "/data/set_train/"
Test_Data_Path = Code_Location + "/data/set_test/"

Train_Target_File_Path = Code_Location + "targets.csv"
Test_Target_File_Path = Code_Location + "submission.csv"

######################################################################	
# 						END (important node)						 #
######################################################################


######################################################################
# 					Variables agnostic to custom setup 			 	 #
######################################################################

File_Extension = ".nii"

TRAIN_SAMPLES = 278
TEST_SAMPLES = 138

# Number was assembled based on comparison 
# between squared error on the different feature number
Number_Of_Features = 355000

Precomputed_Directory = 'PrecomputedData'
Preprocessed_Train_Input_File = Precomputed_Directory + '/train_input.pkl'
Preprocessed_Test_Input_File = Precomputed_Directory + '/test_input.pkl'
Preprocessed_Mask_File = Precomputed_Directory + '/mask.pkl'
Preprocessed_Features_File = Precomputed_Directory + '/feature_selection.pkl'