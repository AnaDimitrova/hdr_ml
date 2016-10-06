import os
import numpy as np
import nibabel as nib

from numpy import *
from nibabel.testing import data_path

import ReadData as rd
import Constants as const

def SplitData():
	data_from_all_files = rd.GetAllFiles()

	startCoordinates = []
	for x in np.arange(0, const.Length_X, (const.Length_X / const.Level_Separation_X)):
		for y in np.arange(0, const.Length_Y, (const.Length_Y / const.Level_Separation_Y)):
			startCoordinates.append([x,y])

	print "Reading files - this may take a bit...";
	
	all_pictures_split_data = []

	for data_from_file in data_from_all_files:
		picture = data_from_file.get_data()
		picture_split_data = []
		for subpicture in picture: # 176 sub pictures
			split_picture = []
			for coordinates in startCoordinates: # for each subpicture split based on coordinates
				x = coordinates[0];
				y = coordinates[1];
				split_picture.append(subpicture[x: x + const.Length_X / const.Level_Separation_X, y:y + const.Length_Y / const.Level_Separation_Y ])

			picture_split_data.append(split_picture)

		all_pictures_split_data.append(picture_split_data);

	print "Done reading files."
	return all_pictures_split_data;