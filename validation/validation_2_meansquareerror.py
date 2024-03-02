import pandas as pd
import numpy as np
import xgboost as xg
from sklearn import metrics
import os
from os import path
import csv
# p = "/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/sample/mean_square_error.csv"
p = "C:/Ronald/uOttawa/CSI 6900/Metallic-main/exp_validation/sample/xgb_mean_square_error.csv"
l = path.exists(p)
fields = ['Test_dataset', 'Classifier', 'F1', 'G-mean', 'Accuracy', 'Precision', 'Recall','ROC_AUC','PR_AUC','Balanced_Accuracy','CWA']
if l == True:
    os.remove(p)
write_file = open(p, 'w', newline='')
csvwriter = csv.writer(write_file)
csvwriter.writerow(fields)
# df = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Metafeature/features.csv")
df = pd.read_csv("C:/Ronald/uOttawa/CSI 6900/Metallic-main/Metafeature/features.csv")
dataset_names=list(df['Dataset'])
names=np.unique(dataset_names)
total_metric=[]
for j in ['KNN','DT','GNB','SVM','RF','GB','ADA','CAT']:
    for i in names:
        total_metric=[]
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
            total_metric.append(metrics.mean_squared_error(y_test, y_pred))
        print(i,j,total_metric)

        csvwriter.writerow([i,j,total_metric[0],total_metric[1],total_metric[2],total_metric[3],total_metric[4], total_metric[5], total_metric[6], total_metric[7], total_metric[8]])
write_file.close()
