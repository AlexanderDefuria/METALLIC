import pandas as pd
import numpy as np
from sklearn import metrics
import os
from os import path
import csv
import sklearn
import krippendorff
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
# p = "/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/deep_scores/kappa_score.csv"
# p1="/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/deep_scores/krippendorf.csv"
# p2="/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/deep_scores/precission.csv"
p = "C:/Ronald/uOttawa/CSI 6900/Metallic-main/exp_validation/deep_scores3/kappa_score.csv"
p1 = "C:/Ronald/uOttawa/CSI 6900/Metallic-main/exp_validation/deep_scores3/krippendorf.csv"
p2 = "C:/Ronald/uOttawa/CSI 6900/Metallic-main/exp_validation/deep_scores3/precision.csv"
p3 = "C:/Ronald/uOttawa/CSI 6900/Metallic-main/exp_validation/deep_scores3/nn_mse.csv"
l = path.exists(p)
l1=path.exists(p1)
l2=path.exists(p2)
l3=path.exists(p3)
fields = ['Test_dataset', 'Classifier', 'F1', 'G-mean', 'Accuracy', 'Precision', 'Recall','ROC_AUC','PR_AUC','Balanced_Accuracy','CWA']
if l == True:
    os.remove(p)
if l1 == True:
    os.remove(p1)
if l2 == True:
    os.remove(p2)
if l3 == True:
    os.remove(p3)
write_file = open(p, 'w', newline='')
write_file1=open(p1,'w',newline='')
write_file2=open(p2,'w',newline='')
write_file3=open(p3,'w',newline='')
csvwriter = csv.writer(write_file)
csvwriter1 = csv.writer(write_file1)
csvwriter2 = csv.writer(write_file2)
csvwriter3 = csv.writer(write_file3)
csvwriter.writerow(fields)
csvwriter1.writerow(fields)
csvwriter2.writerow(fields)
csvwriter3.writerow(fields)
# df = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Metafeature/features.csv")
df = pd.read_csv("C:/Ronald/uOttawa/CSI 6900/Metallic-main/Metafeature/features.csv")
dataset_names=list(df['Dataset'])
names=np.unique(dataset_names)
kapa_score=[]
krippen_score=[]
precision_5=[]
total_metric=[]
# scaler = MinMaxScaler()
scaler = StandardScaler()
for j in ['KNN','DT','GNB','SVM','RF','GB','ADA','CAT']:
    for i in names:
        print(i)
        kapa_score=[]
        krippen_score=[]
        precision_5=[]
        total_metric=[]
        for metric in ['F1','G-mean','Accuracy','Precision','Recall','AUC-ROC','AUC-PR','BalancedAccuracy','CWA']:
            dataframe = df[df['Dataset'] != i]
            x_test_sample = df[df['Dataset'] == i]
            x_test_selected = x_test_sample[x_test_sample[j] == 1]
            y_test = np.array(x_test_selected[metric])
            x_test = x_test_selected.iloc[:, 1:50]
            # x_test = x_test_selected.iloc[:, 1:49]
            x_test_sample=x_test.iloc[:,28:]
            # x_test_sample=x_test.iloc[:,27:]

            count1=1
            sample_not_present=[]
            for kk in range(0,21):
                sample1=list(x_test_sample.iloc[:,kk])
                if 1 not in sample1:
                    sample_not_present.append(count1)
                count1=count1+1
            selected_dataframe = dataframe[dataframe[j] == 1]
            y_train = np.array(selected_dataframe[metric])
            x_train = selected_dataframe.iloc[:, 1:50]

            x_train = scaler.fit_transform(np.array(x_train))
            x_test = scaler.transform(np.array(x_test))

            model = Sequential()
            model.add(Dense(64, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.4))
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
            model.add(Dense(1))
            # compile the keras model
            model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['MeanSquaredError'])

            # early_stopping = EarlyStopping(monitor='mean_squared_error', patience=5)
            # fit the keras model on the dataset
            # x_train=np.array(x_train)
            # y_train[np.isnan(y_train)] = 0
            # model.fit(np.array(x_train),y_train, epochs = 5, callbacks=[early_stopping], verbose = 2)
            # model.fit(np.array(x_train),y_train, epochs = 5, verbose = 2)
            model.fit(x_train,y_train, epochs = 5, verbose = 0)

            y_pred = list(model.predict(np.array(x_test)))
            y_preds=[]
            for item in y_pred:
                y_preds.append(item[0])
            if len(y_preds) != 21:
                for ii in sample_not_present:
                    y_preds.insert(int(ii)-1,0)
            y_pred = pd.DataFrame(y_preds)
            ranking = y_pred.rank(ascending=False,method="first")
            ranking = np.array(ranking)
            pred_ranking=[]
            for ij in ranking:
                for jj in ij:
                    pred_ranking.append(jj)
            y_test=list(y_test)
            if len(y_test)!=21:
                for ij in sample_not_present:
                    y_test.insert(int(ij)-1,0)
            y_test=pd.DataFrame(y_test)

            ranking1 = y_test.rank(ascending=False,method="first")
            ranking1 = np.array(ranking1)
            real_ranking = []
            for ij in ranking1:
                for jj in ij:
                    real_ranking.append(jj)
            print(real_ranking)
            print(pred_ranking)
            real_ranking = [0 if math.isnan(x) else x for x in real_ranking]
            pred_ranking = [0 if math.isnan(x) else x for x in pred_ranking]
            kapa_score.append(sklearn.metrics.cohen_kappa_score(np.array(real_ranking),np.array(pred_ranking)))

            reliability_data=[list(real_ranking),list(pred_ranking)]
            krippen_score.append(krippendorff.alpha(reliability_data=reliability_data))
            real_5=[]
            pred_5=[]
            for r in range(1,6):
                if r in real_ranking:
                    index=real_ranking.index(r)
                    real_5.append(index+1)
                if r in pred_ranking:
                    index1=pred_ranking.index(r)
                    pred_5.append(index1+1)
            count=0
            for rk in pred_5:
                if rk in real_5:
                    count=count+1
            precision_5.append(count/5)

            total_metric.append(metrics.mean_squared_error(y_test, y_pred))
            # print(i,j,total_metric)

        csvwriter.writerow([i, j, kapa_score[0], kapa_score[1], kapa_score[2], kapa_score[3], kapa_score[4], kapa_score[5], kapa_score[6], kapa_score[7], kapa_score[8]])
        csvwriter1.writerow([i, j, krippen_score[0], krippen_score[1], krippen_score[2], krippen_score[3], krippen_score[4], krippen_score[5], krippen_score[6], krippen_score[7], krippen_score[8]])
        csvwriter2.writerow([i, j, precision_5[0], precision_5[1], precision_5[2], precision_5[3], precision_5[4],precision_5[5], precision_5[6], precision_5[7], precision_5[8]])
        csvwriter3.writerow([i, j, total_metric[0],total_metric[1],total_metric[2],total_metric[3],total_metric[4], total_metric[5], total_metric[6], total_metric[7], total_metric[8]])
write_file.close()
write_file1.close()
write_file2.close()
write_file3.close()
