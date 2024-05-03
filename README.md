# METALLIC
Metalearning for Tackling the Class Imbalance Problem
Dataset folder contains dataset files. 
Creating_metafeatures folder contains all the files related to the creation of metafeatures and one single file is created in metafeature folder as output file.To start execution start with main.py.
Recommendation folder contains all the files required to run the recommendation system and to start recommendation system user need to give the name of file on which he needs to test, the select the metrics and base classifier.Use test2.py to start execution.
Validation folder contains files of validating of our model.

# How to run the code

# Dataset
It contains all the 155 datasets.

# MetaFeature folder
The "feature.csv" file contains the meta features.

# creating meta feature
If you add more "ariff" format data, store them to "additional_dataset" folder first, then head to the "processing_additional_data.ipynb" and use it to transform data into ".csv" file and store them to the "processed_dataset" folder. You may have to change their file name manually because sometime the name does not look nice. 

If you want to add more models, metrics, or resampling strategies, wrap them in a ".py" file, and import them in the "main.py" to use.

Run the "main.py" to get the meta-feature

The "test_dataset" here is to test the recommendation system. Please **do not** use them in "creating_metafeatures" part and any training process.

"data_statistics.ipynb" is used for summarizing all the dataset

# validation
Models are already built but notice that neural network models need scaling to get a better result. Otherwise, the MSE would be extremely unstable. "deep_validation.py" is not a good example because it does not have scaling process. Please use "deep_validation2.py" and "deep_validation3.py"

There are four notebooks for comparing the result. You can use them to create comparison tables and specific plots.

For "comparison_score_for_*" files, I just copied the data from the tables from "comparison-2" file, which is convenient and then plot the result.

The comparison_plot_all is based on all the datasets and we take F1 score for example.

# exp_validation
This folder contain the results of four meta-learning algorithms.

# Recommendation System
"final_model.py" is the model for backend recommendation system.

"app.py" is the server that connect the back end with the front end.

"test_dataset" folder here is just for temporarily saving.

I also copied it on the root because of the cloud server requirement.

# my-resampling-strategy-app
"ResamplingStrategyForm.js" is the code designed for the front end web page. "App.css" is the layout style. So you can modifiy it based on what you need.

