import pandas as pd
import numpy as np
import xgboost as xg
from sklearn import metrics
import os
from os import path
import csv
import joblib

fields = ['Test_dataset', 'Classifier', 'F1', 'G-mean', 'Accuracy', 'Precision', 'Recall','ROC_AUC','PR_AUC','Balanced_Accuracy','CWA']

df = pd.read_csv("./Metafeature/features.csv")
dataset_names=list(df['Dataset'])
names=np.unique(dataset_names)
total_metric=[]
for j in ['KNN','DT','GNB','SVM','RF','GB','ADA','CAT']:
    for i in names:
        for metric in ['F1','G-mean','Accuracy','Precision','Recall','AUC-ROC','AUC-PR','BalancedAccuracy','CWA']:
            dataframe = df[df['Dataset'] != i]
            x_test_sample = df[df['Dataset'] == i]
            x_test_selected = x_test_sample[x_test_sample[j] == 1]
            y_test = np.array(x_test_selected[metric])
            # x_test = x_test_selected.iloc[:, 1:49]
            x_test = x_test_selected.iloc[:, 1:50]
            selected_dataframe = dataframe[dataframe[j] == 1]
            y_train = np.array(selected_dataframe[metric])
            # x_train = selected_dataframe.iloc[:, 1:49]
            x_train = selected_dataframe.iloc[:, 1:50]
            model = xg.XGBRegressor(objective ='reg:squarederror',colsample_bytree=0.4, gamma=0, learning_rate=0.07, max_depth=3,
                                                min_child_weight=1.5,n_estimators=10000, reg_alpha=0.75, reg_lambda=0.45, subsample=0.6, seed=42)
                        
            x_train = np.array(x_train)
            y_train[np.isnan(y_train)] = 0
            model.fit(np.array(x_train), y_train)
            y_pred = model.predict(np.array(x_test))
            # print(len(y_pred))
            y_test[np.isnan(y_test)] = 0
            y_pred[np.isnan(y_pred)] = 0

            # Save the XGB model to a file
            model_filename = f"xgb_model_{j}_{metric}.pkl"
            joblib.dump(model, model_filename)
            





