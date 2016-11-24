# Team Name
HDR

# Team Members
Ana Dimitrova (nana@student.ethz.ch)
Ines Haymann (haymanni@student.ethz.ch)
Jonathan Rotsztejn (rotsztej@student.ethz.ch)


# Description Project 2
1. Data Preprocessing
Learning from the experience shared from other teams after the first project we decided to clean up more our data and create a real set of features instead of relying mostly on SelectKBest.
	-> Clean the "black" areas in the files in two ways:
		1) For each nii file we have 206 different pictures. Approximately the first and last 30 of them have mostly black background and not much of the brain, so we filtered them out.
		2) For each of the remaining 146 pictures, we cleaned up the corners of the data, removing a 40 pixel margin from the sides of the picture.
	-> Split the data into cubes - We split the now clean from noise data into 9 equally sized cubes. 
	-> Reading and processing of the data was still very time consuming, so we changed our approach to preprocess each one of the files individually and then keep only the cleaned up data in separate split files. This removed the constraint to be able to fit all the data into memory to run the project.

2. Select K Features
For each of the files and each of the cubes we extracted 2 sets of features:
	1) The mean, standard deviation and median of the points in the cube
	2) Histograms on the cubes. Two important points here, we aimed to have a reasonably partitioned bins where the difference in size wouldn't be too big. So we ended up with 1000 bins. Also we filtered any remaining black pixels from the histograms, as they don't add any value. That resulted was 999 bins.
In total we created 9 * (3 + 999) ~ 9000  features. To select the best of them we used sklearn SelectKBest. To arrive at our most performant feature selection we tried selecting different amount of features but at 55 we achieved the best results.

3. Training the model. We tried different algorithms but the best and most performant one was Lasso with cross validation with number of folds equal to 10. 

# Description Project 1
1. Data Preprocessing 
The input data was in nii format. There are couple of major drawbacks we overcame:
	-> Each nii image has some amount of data that is not relevant to the brain properties i.e. the background of the image. Each one of the files was preprocessed to exclude that data. For that purpose, a simple mask was created to clean the background data and make the images uniform.

	-> Reading the data was very time consuming. To optimize that, we had the data preprocessed one time in a specific format and then reused it multiple times to find the best model.

2. Select K features 
This process was likely the most time-consuming part. 
We initially worked towards finding features that would qualify/correlate well with brain age. Based on some research we did, we found out that the color of the brain and width of brain wall should be good approximators for brain age. Our team worked on producing these features and show their correlation with histogram of our training set and the features. However, we couldn't produce enough relevant features to build a good model. Thus, we used the sklearn SelectKBest features method. 

To find the right number of features that would give us the best model, we used a script that compared the Squared Error for the training set - using cross-validation to see which one produced the smallest error. 

3. Training the model
The number of features that turned out to be most efficient ended up being quite big (355,000). Thus, we wanted to minimize and potentially eliminate some of the features that were not as necessary. Therefore we initially started predicting using Lasso regression. However, later, Ridge regression showed much better results. We also compared it to Linear regression, ElasticNet (mix between Ridge and Lasso), BayesianRidge, BaggingRegressor, AdaBoostRegressor and DecisionTreeRegressor. 
The ridge regression is executed in combination with s Leave-One-Out cross-validation. Here, we experimented with different types of cross-validations. Leave-One-Out produced best results.

4. Postprocessing
As a post-processing step, we used our prior knowledge of the distribution of the training set's ages to improve the precision of our results by adapting it lightly to get a similar age distribution. This step is randomizing since it draws samples randomly from the original distribution and, therefore, can produce different results after each execution.
Note: The postprocessing step is not part of predict_final.py as we wanted to include only the deterministic part of our processing in the script. We did that to improving separately the deterministic algorithm and then once we found the best approach we would apply the postprocessing. It however can be found in ShapeOutput.py.

# Running the project
1. Make sure you create a global variable \"ML_PATH\" for your project location in your ~/.bash_profile, or update Constants.py with the location you have unzipped the code.
2. Make sure that you have the training and test data in a  "data" folder in the code directory, or update the Constants.py file with the location of the data on your machine. Same applies to targets.csv and submissions.csv






