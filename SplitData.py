import os
import numpy as np
import nibabel as nib

from numpy import *
from nibabel.testing import data_path
from sklearn.externals import joblib

import ReadData as rd
import Constants as const
import ProcessData as pd

# TODO: Need to decide how many splits should we filter out
SPLIT_MARGIN = 30;
# TODO: Need to decide how many pixels do we crop from the sides of each split
PIXELS_CROP_FROM_SIDE = 40;
# TODO: Need to decide how many cubes do we want to split the data into
SEGMENT_COUNT_X = 3
SEGMENT_COUNT_Y = 3

def RemoveMarginalSplits(data):
	return data[SPLIT_MARGIN:len(data) - SPLIT_MARGIN];

def CropBlackArea(data):
	maxPixels1 = len(data[0]) - PIXELS_CROP_FROM_SIDE;
	maxPixels2 = len(data[0][0]) - PIXELS_CROP_FROM_SIDE;
	return data[:,PIXELS_CROP_FROM_SIDE:maxPixels1,PIXELS_CROP_FROM_SIDE:maxPixels2]


def ProcessFiles(filesLocation, outputDirectory, preprocess):
	files = rd.GetFileNames(filesLocation);
	if(not preprocess and len(files) == 0):
		print(bcolors.WARNING + "Warning: No active frommets remain. Continue?" + bcolors.ENDC)

	if(preprocess):
		files = rd.GetFileNames(filesLocation);
		a = 0;
		for path in files:
			img = nib.load(path)
			croppedData = CropDataFromFile(img);
			joblib.dump(croppedData, outputDirectory + str(a) + '.pkl');
			a +=1 

def CropDataFromFile(file):	
	# Remove the splits of the files that are mostly black
	data = RemoveMarginalSplits(file.get_data());
	# Remove the black area of the sides of each slice
	data = CropBlackArea(data);
	# Split the data into cubes
	output = SplitIntoCubes(data);
	
	return output

def SplitIntoCubes(data):
	segmentSizeX = int(floor(len(data[0])/SEGMENT_COUNT_X));
	segmentSizeY = int(floor(len(data[0][0])/SEGMENT_COUNT_Y));
	remainderX = len(data[0])%segmentSizeX
	remainderY = len(data[0][0])%segmentSizeY
	
	xSplits = range(remainderX, len(data[0]), segmentSizeX);
	ySplits = range(remainderY, len(data[0][0]), segmentSizeY);
	
	output = []	
	for x in xSplits:
		for y in ySplits:
			cube = data[:, x:(x+segmentSizeX), y:(y+segmentSizeY)]
			output.append(cube)

	return output;

def ExtractFeaturesFromFile(data):
	features = []

	for cube in data:
		cube = cube.tolist()
		features.append(mean(cube))
		features.append(std(cube));
		features.append(median(cube));

	return features;
	

def ExtractFeaturesFromAllFiles(inputDirectory, outputFile, preprocess):
	if(not preprocess and not(os.path.isfile(outputFile))):
		print(bcolors.WARNING + "Warning: No active frommets remain. Continue?" + bcolors.ENDC)

	features = []
	if(preprocess):
		files = rd.GetFileNames(inputDirectory, 'pkl');
		a = 0;
		for file in files:
			input = joblib.load(file);
			features.append(ExtractFeaturesFromFile(input));
			print (a)
			a += 1

		joblib.dump(features, outputFile);
	else:
		features = joblib.load(outputFile);
	
	return features;







