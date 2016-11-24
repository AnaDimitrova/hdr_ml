# Team Name
HDR

# Team Members
Ana Dimitrova (nana@student.ethz.ch)
Ines Haymann (haymanni@student.ethz.ch)
Jonathan Rotsztejn (rotsztej@student.ethz.ch)

# Description Project 2
1. Data Preprocessing
Learning from the experience shared from other teams after the first project we decided to clean up more our data and create a smaller set of better defined features instead of relying mostly on SelectKBest.
	-> Clean the "black" areas in the files in two ways:
		1) For each nii file we have 206 different pictures. Approximately the first and last 30 of them have mostly black background and not much of the brain, so we filtered them out.
		2) For each of the remaining 146 pictures, we cleaned up the corners of the data, removing a 40 pixel margin from each of the sides of the picture.
	-> Split the data into cubes - We split the now clean from noise data into 9 equally sized cubes. 
	-> Reading and processing of the data was still very time consuming, so we changed our approach to preprocess each one of the files individually and then keep only the cleaned up data in separate files. This reduced a lot our process time since we had to fit a lot less data in memory to run the project.

2. Select K Features
For each of the files and each of the cubes we extracted 2 sets of features:
	1) The mean, standard deviation and median of the points in the cube
	2) Histograms on the cubes. We aimed to have a reasonably partitioned bins where the difference in size wouldn't be too big getting as close as possible to uniform distribution. So we ended up with 1000 bins. Also we filtered any remaining black pixels from the histograms, as they don't add any value. That resulted was 999 bins.
In total that created 9 * (3 + 999) ~ 9000  features. To select the best of them we used sklearn SelectKBest. To arrive at our most performant feature selection we tried selecting different amount of features but at 60 we achieved the best results.

3. Training the model. 
We tried different algorithms but the best and most performant one was Lasso with cross validation with number of folds equal to 20. 

4. Post processing
We limited the result data such that we don't output 1 and 0 as the effect of a wrong 0 or 1 has a strong effect on the final score. Thus we replaces all 1's and O's with 0.99 and 0.01 respectively.
We were able to get quite good results on the visible training set (0.21698) by boosting our predictions when they are close to 1 and 0, by making them even closer to the boundaries. However for our final submission we decided it will be better to avoid that approach since the rick of misjudged element in the hidden training set would have been seriously detrimental in our final score.

# Running the project
1. Make sure you create a global variable \"ML_PATH\" for your project location in your ~/.bash_profile, or update Constants.py with the location you have unzipped the code.
2. Make sure that you have the training and test data in a  "data" folder in the code directory, or update the Constants.py file with the location of the data on your machine. Same applies to targets.csv and submissions.csv
