import os
import numpy as np
import nibabel as nib
import Constants as const
import re

from nibabel.testing import data_path
from os import listdir
from os.path import isfile, join	

# Get all the files that are in the passed directory
def GetAllFiles(dataPath = const.Test_Data_Path):
	testFileNames = [f for f in listdir(dataPath) if isfile(join(dataPath, f)) and f.endswith(const.File_Extension)]
	testFileNames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
	loadedFiles = []

	for fileName in testFileNames:

		path = os.path.join(dataPath, fileName)
		img = nib.load(path)
		loadedFiles.append(img)

	return loadedFiles



