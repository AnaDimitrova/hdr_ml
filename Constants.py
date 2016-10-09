
import os

if(not os.environ.has_key("ML_PATH")):
	raise Exception("Make sure you create a global variable \"ML_PATH\" for your project location in your ~/.bash_profile.")
#Enlistment_Location = "/Users/jr/Desktop/Machine Learning/"

Enlistment_Location = os.environ['ML_PATH']

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