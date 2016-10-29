
import os

if(not "ML_PATH" in os.environ):
	raise Exception("Make sure you create a global variable \"ML_PATH\" for your project location in your ~/.bash_profile, or update Constants.py")

Enlistment_Location = os.environ['ML_PATH']
# You can specify the enlistment location here. Example below:
#Enlistment_Location = "/Users/jr/ETH/Machine Learning/"

Train_Data_Path = Enlistment_Location + "/data/set_train/"
Test_Data_Path = Enlistment_Location + "/data/set_test/"

File_Extension = ".nii"

Train_Target_File_Path = Enlistment_Location + "targets.csv"
Test_Target_File_Path = Enlistment_Location + "submission.csv"

Level_Separation_X = 4
Level_Separation_Y = 4

Length_X = 208
Length_Y = 176

TRAIN_SAMPLES = 278
TEST_SAMPLES = 138

# Number was assembled based on comparison 
# between squared error on the different feature number
Number_Of_Features = 355000

Preprocessed_Train_Input_File = 'PrecomputedData/train_input.pkl'
Preprocessed_Test_Input_File = 'PrecomputedData/test_input.pkl'
Preprocessed_Mask_File = 'PrecomputedData/mask.pkl'
Preprocessed_Features_File = 'PrecomputedData/feature_selection.pkl'