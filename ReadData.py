import os
import numpy as np
import nibabel as nib
import Constants as const
import re

from nibabel.testing import data_path
from os import listdir
from os.path import isfile, join	

def GetFileNames(dataPath = const.Test_Data_Path, fileExtension = const.File_Extension):
	testFileNames = [f for f in listdir(dataPath) if isfile(join(dataPath, f)) and f.endswith(fileExtension)]
	testFileNames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
	files = []
	for fileName in testFileNames:

		files.append(os.path.join(dataPath, fileName))
	return files

# Get all the files that are in the passed directory
def GetAllFiles(dataPath = const.Test_Data_Path):
	files = GetFileNames(dataPath);
	loadedFiles = []

	for path in files:
		img = nib.load(path)
		loadedFiles.append(img)

	return loadedFiles



