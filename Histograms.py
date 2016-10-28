import nibabel
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import BaggingRegressor

import csv

import ReadData as rd
import Constants as const

import matplotlib.pyplot as plt


def applyMask(imageData, mask):

	#Remove background (perhaps unnecessary)
	noBlack = imageData[:, :, 110, 0][ mask ]

	return noBlack

# data for one file turned into an array after a mask has been applied
def getFieldNames():
	histogramMin = 100;
	# the NII file boundaries are not explicitly set for these files.
	# This is the approximate value found in the files we have
	histogramMax = 1000;
	binSize = 100; 

	fieldnames = []
	fieldnames.append('fileIndex');
	for i in range(histogramMin, histogramMax, binSize):
		fieldnames.append('more than ' + str(i) + ' pixel color');
		fieldnames.append('between than ' + str(i) + ' and ' + str(i + 100) + ' pixel color');

	fieldnames.append('age')

	return fieldnames

# data for one file turned into an array after a mask has been applied
def collectHistogramData(fileIndex, rawData):
	histogramMin = 100;
	# the NII file boundaries are not explicitly set for these files.
	# This is the approximate value found in the files we have
	histogramMax = 1000;
	binSize = 100; 

	result = {'fileIndex':fileIndex}
	for i in range(histogramMin, histogramMax, binSize):
		result['more than ' + str(i) + ' pixel color'] = len(np.where(rawData>i)[0]);
		result['between than ' + str(i) + ' and ' + str(i + 100) + ' pixel color'] = len(np.where(abs(rawData - i - 50)<50)[0]);
	return result 

def generateInputFor(path, numberOfCases, mask):
	
	#Read image files
	data = rd.GetAllFiles(path)

	#Calculate mask to remove background from first input
	if(mask == None):
		mask = np.where(data[0].get_data()[:, :, 110, 0]>0)

	histograms = []
	fieldnames = []
	for i in range(1,numberOfCases):
	#for i in range(1,3):
		#print(i)
		processedImage = applyMask(data[i].get_data(), mask)
		histograms.append(collectHistogramData(i, processedImage));

	return {'histograms': histograms}


def processDataForHistograms():
	result = generateInputFor(const.Train_Data_Path, const.TRAIN_SAMPLES, None)
	histograms = result['histograms']
	fieldnames = getFieldNames()

	output = np.genfromtxt(const.Train_Target_File_Path, delimiter='\n')
	for histogram, age in zip(histograms, output):	
		histogram['age'] = age

	with open('histograms.csv', 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

		writer.writeheader()
		for histogram in histograms:
			writer.writerow(histogram);

	return histograms;


histograms = processDataForHistograms();

fieldnames = getFieldNames()
# data = {}
# with open('histograms.csv') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
# 	    for name in fieldnames:
# 	    	if(name in data):
# 	    		data[name].append(row[name])

# 	    	if(name not in data):
# 	    		data[name] = [row[name]]


uniqueAges = {}
for name in fieldnames:
	allValues = {}
	if(name != 'age' and name != 'fileIndex'):
		for histogram in histograms:
			currentAge = histogram['age'];
			if currentAge in allValues:
				allValues[currentAge].append(histogram[name])
			else:
				allValues[currentAge] = [histogram[name]]

		y = []
		ages = list(allValues.keys())
		for age in ages:
			# print (allValues[age])
			average =int(sum(allValues[age]) / len(allValues.keys())); 
			y.append(average)

		# print (y)
		# print (ages)
		plt.plot(ages, y, alpha=0.5)
		plt.title('Plot with ' + name)
		plt.xlabel('Age')
		plt.ylabel('Number Of Pixels')
		plt.show()

# print(histograms)






















