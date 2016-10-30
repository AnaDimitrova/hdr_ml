# Team Name
HDR

# Team Members
Ana Dimitrova (nana@student.ethz.ch)
Ines Haymann (haymanni@student.ethz.ch)
Jonathan Rotsztejn (rotsztej@student.ethz.ch)

# Description
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






