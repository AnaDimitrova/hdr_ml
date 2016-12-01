import os
import numpy as np
import nibabel as nib

from numpy import *
from nibabel.testing import data_path
from sklearn.externals import joblib

import ReadData as rd
import Constants as const
import ProcessData as pd

# Remove first and last few slices from each nii file as they contain mostly black dots.
def RemoveMarginalSplits(data):
	return data[const.SPLIT_MARGIN:len(data) - const.SPLIT_MARGIN];

# Crop margin from each slice of a nii file, ot clean up data set.
def CropBlackArea(data):
	maxPixels1 = len(data[0]) - const.PIXELS_CROP_FROM_SIDE;
	maxPixels2 = len(data[0][0]) - const.PIXELS_CROP_FROM_SIDE;
	return data[:, const.PIXELS_CROP_FROM_SIDE:maxPixels1, const.PIXELS_CROP_FROM_SIDE:maxPixels2]


# Process one by one the files for the all prediction to be able to be handles by a regular pc
def ProcessFiles(filesLocation, outputDirectory, preprocess):
	if not(os.path.exists(outputDirectory)):
		os.makedirs(outputDirectory)

	files = rd.GetFileNames(filesLocation);
	
	if(not preprocess and len(files) == 0):
		print(bcolors.WARNING + "Warning: You should first create the files by changing varibales Preprocess_Train_Files." + bcolors.ENDC)

	if(preprocess):
		files = rd.GetFileNames(filesLocation);
		a = 0;
		for path in files:
			img = nib.load(path)
			croppedData = CropDataFromFile(img);
			joblib.dump(croppedData, outputDirectory + str(a) + '.pkl');
			a +=1 

# Split the data in each one of the nii files into cubes
def SplitIntoCubes(data):
	segmentSizeX = int(floor(len(data[0])/const.SEGMENT_COUNT_X));
	segmentSizeY = int(floor(len(data[0][0])/const.SEGMENT_COUNT_Y));
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

# General function to clean up a file from the background and split it into cubes
def CropDataFromFile(file):	
	# Remove the splits of the files that are mostly black
	data = RemoveMarginalSplits(file.get_data());
	# Remove the black area of the sides of each slice
	data = CropBlackArea(data);
	# Split the data into cubes
	output = SplitIntoCubes(data);
	
	return output

# Extract the features from each one of the files.
def ExtractFeaturesFromFile(data, bins):
	features = []

	for cube in data:
		features.append(np.mean(cube))
		features.append(np.std(cube));
		features.append(np.median(cube));
		histogram = np.histogram(cube, bins=bins);
		# Remove the black information
		features.extend(histogram[0][1:]);

	return features;
	
# Builds a set of histograms. Cleans up outliers so that we have the data somewhat uniformly distributed
def BuildHistogramBins(): 
	files = rd.GetFileNames(const.Precomputed_Train_Directory, 'pkl');
	input = joblib.load(files[0]);
	histogram = np.histogram(input[0], bins=const.NUMBER_OF_BINS);
	trimmedBins = []
	for h in zip(histogram[0], histogram[1]):
		# Remove outliers
		if(h[0] >= 100):
			trimmedBins.append(h[1])

	return trimmedBins;

# Extract features from each one of the files which have already been preprocessed.
def ExtractFeaturesFromAllFiles(inputDirectory, outputFile, bins, preprocess):
	if(not preprocess and not(os.path.isfile(outputFile))):
		print(bcolors.WARNING + "Warning: You should first create the files by changing varibales Preprocess_Train_Files" + bcolors.ENDC)

	features = []
	if(preprocess):
		files = rd.GetFileNames(inputDirectory, 'pkl');
		a = 0;
		for file in files:
			input = joblib.load(file);
			features.append(ExtractFeaturesFromFile(input, bins));
			print (a)
			a += 1

		joblib.dump(features, outputFile);
	else:
		features = joblib.load(outputFile);
	
	return features;







