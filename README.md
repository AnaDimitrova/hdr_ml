# Team Name
HDR

# Team Members
Ana Dimitrova (nana@students.ethz.ch)
Ines Haymann (haymanni@students.ethz.ch)
Jonathan Rotsztejn (rotsztej@students.ethz.ch)

# Description
1. Data Preprocessing 
The input data was in nii format. There are couple of major drawbacks we overcame:
	-> Each nii image has bit amount of data that not relevant to the brain properties i.e. the background of the image. Each one of the files was preprocessed to exclude that data. For that purpose a simple mask was created to clean the background data and make the images uniform.

	-> Reading the data was very time consuming. To optimize that we had the data preprocessed one time in a specific format and then reused mutliple times to find the best model.

2. Select K features. 
This process was likely the most time consuming part. 
We initially worked towards finding features that would qualify/correlate well with brain age. Based on some research we did we found out that the color of the brain and width of brain wall should be good approximators for brain age. Our team worked on producing these features and show their correlation with histogram of our training set and the features, however we couldn't produce enough relevant features to build a good model. Thus we used the sklearn SelectKBest features method. 

To find the right number of features that would give us best model we used a script that compared the Squared Error for out training set - using cross validation on the training set to see which produces the smallest error. 

3. Training the model.
The number of features that turned out to be most efficient end up being quite big (355000). Thus we wanted to minimize and potentially eliminate some of the features that are not as necessary. Therefore we initially started approximating using Lasso regression. However later Ridge regression showed much better results. We also compared it to Linear regression, ElasticNet (mix between Ridge and Lasso), BayesianRidge, BaggingRegressor, AdaBoostRegressor and DecisionTreeRegressor. 
The ridge regression is executed with combination with Leave-One-Out cross-validation. Here we experimented with different types of cross validations, however again the Leave-One-Out produced best results.

4. Postprocessing.
As a post processing step we approximated our results to the initial input set to get similar age distribution. 


