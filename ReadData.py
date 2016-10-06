import os
import numpy as np
import nibabel as nib
import Constants as const

from nibabel.testing import data_path
from os import listdir
from os.path import isfile, join	

def GetAllFiles(dataPath = const.Test_Data_Path):
	testFileNames = [f for f in listdir(dataPath) if isfile(join(dataPath, f)) and f.endswith(const.File_Extension)]
	loadedFiles = []

	for fileName in testFileNames:

		path = os.path.join(dataPath, fileName)
		img = nib.load(path)
		loadedFiles.append(img)

	return loadedFiles



