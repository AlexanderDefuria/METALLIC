import pandas as pd
import numpy as np
from sklearn import metrics
import os
from os import path
import csv
import sklearn
import krippendorff
import sys
# sys.path.insert(1, "/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Recommendation_system")
sys.path.insert(1, "C:/Ronald/uOttawa/CSI 6900/Metallic-main/Recommendation_system")
import knn_algo
# from Recommendation_system import knn_algo
import math
# p = "/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/knn_scores/kappa_score.csv"
# p1="/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/knn_scores/krippendorf.csv"
# p2="/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/knn_scores/precission.csv"
p = "C:/Ronald/uOttawa/CSI 6900/Metallic-main/exp_validation/knn_scores/kappa_score.csv"
p1 = "C:/Ronald/uOttawa/CSI 6900/Metallic-main/exp_validation/knn_scores/krippendorf.csv"
p2 = "C:/Ronald/uOttawa/CSI 6900/Metallic-main/exp_validation/knn_scores/precision.csv"
l = path.exists(p)
l1=path.exists(p1)
l2=path.exists(p2)
fields = ['Test_dataset', 'Classifier', 'F1', 'G-mean', 'Accuracy', 'Precision', 'Recall', 'ROC_AUC','PR_AUC','Balanced_Accuracy','CWA']
if l == True:
    os.remove(p)
if l1 == True:
    os.remove(p1)
if l2 == True:
    os.remove(p2)
write_file = open(p, 'w', newline='')
write_file1=open(p1,'w',newline='')
write_file2=open(p2,'w',newline='')
csvwriter = csv.writer(write_file)
csvwriter1 = csv.writer(write_file1)
csvwriter2 = csv.writer(write_file2)
csvwriter.writerow(fields)
csvwriter1.writerow(fields)
csvwriter2.writerow(fields)
# df = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Metafeature/features.csv")
df = pd.read_csv("C:/Ronald/uOttawa/CSI 6900/Metallic-main/Metafeature/features.csv")
dataset_names=list(df['Dataset'])
names=np.unique(dataset_names)
kapa_score=[]
krippen_score=[]
precision_5=[]
imb_strategies = ['None', 'SMOTE', 'NearMiss', 'SMOTEENN', 'Randomoversampling', 'BorderlineSMOTE', 'SVMSMOTE',
            'RandomUnderSampler', 'ClusterCentroids', 'NearMissversion1', 'NearMissversion2', 'NearMissversion3',
            'TomekLinks', 'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours', 'AllKNN',
            'CondensedNearestNeighbour', 'NeighbourhoodCleaningRule', 'InstanceHardnessThreshold', 'SMOTETomek']
# data_c = 0
for j in ['KNN','DT','GNB','SVM','RF','GB','ADA','CAT']:
    for i in names:
        print(i)
        # data_c = data_c +1 
        kapa_score=[]
        krippen_score=[]
        precison_5=[]
        for metric in ['F1','G-mean','Accuracy','Precision','Recall','AUC-ROC','AUC-PR','BalancedAccuracy','CWA']:
            # dataframe = df[df['Dataset'] != i]
            x_test_sample = df[df['Dataset'] == i]
            x_test_selected = x_test_sample[x_test_sample[j] == 1]
            predicted_list = knn_algo.knn_pred_ranking(j, metric, i, df)
            val_list = []
            for k in imb_strategies:
                new_df_real = x_test_sample.loc[x_test_sample[k] == 1]
                new_df_real = new_df_real.loc[new_df_real[j] == 1, metric]
                # print(new_df_real)
                new_df_reallist=list(new_df_real)
                for val in new_df_reallist:
                    value = val
                # value = new_df_real.iloc[0]
                val_list.append(value)
            req = sorted(zip(val_list, imb_strategies), reverse=True)
            # print(req)
            real_list=[]
            real_rank=[]
            pred_rank = []
            for val, imb in req:
                real_list.append(imb)
            for n in range(len(real_list)):
                real_rank.append(n+1)
            for real in real_list:
                if real in predicted_list:
                    pred_rank.append(predicted_list.index(real)+1)
            real_rank = [0 if math.isnan(x) else x for x in real_rank]
            pred_rank = [0 if math.isnan(x) else x for x in pred_rank]
            kapa_score.append(sklearn.metrics.cohen_kappa_score(np.array(real_rank),np.array(pred_rank)))

            reliability_data=[list(real_rank),list(pred_rank)]
            krippen_score.append(krippendorff.alpha(reliability_data=reliability_data))
            # real_5=real_rank[:5]
            # pred_5=pred_rank[:5]
            real_5=real_rank[:5]
            pred_5=pred_rank[:5]
            count=0
            for rk in pred_5:
                if rk in real_5:
                    count=count+1
            precision_5.append(count/5)

            
        
# print(data_c)
        csvwriter.writerow([i, j, kapa_score[0], kapa_score[1], kapa_score[2], kapa_score[3], kapa_score[4], kapa_score[5], kapa_score[6], kapa_score[7], kapa_score[8]])
        csvwriter1.writerow([i, j, krippen_score[0], krippen_score[1], krippen_score[2], krippen_score[3], krippen_score[4], krippen_score[5], krippen_score[6], krippen_score[7], krippen_score[8]])
        csvwriter2.writerow([i, j, precision_5[0], precision_5[1], precision_5[2], precision_5[3], precision_5[4], precision_5[5], precision_5[6], precision_5[7], precision_5[8]])
write_file.close()
write_file1.close()
write_file2.close()

            
            

#             sample_not_present=[]
#             for kk in range(0,21):
#                 sample1=list(x_test_sample.iloc[:,kk])
#                 if 1 not in sample1:
#                     sample_not_present.append(count1)
#                 count1=count1+1
#             selected_dataframe = dataframe[dataframe[j] == 1]
#             y_train = np.array(selected_dataframe[metric])
#             x_train = selected_dataframe.iloc[:, 1:46]
#             model = xg.XGBRegressor(objective ='reg:squarederror',colsample_bytree=0.4, gamma=0, learning_rate=0.07, max_depth=3,
#                                     min_child_weight=1.5,n_estimators=10000, reg_alpha=0.75, reg_lambda=0.45, subsample=0.6, seed=42)
#             x_train = np.array(x_train)
#             model.fit(np.array(x_train), y_train)
#             y_pred = list(model.predict(np.array(x_test)))
#             if len(y_pred) != 21:
#                 for ii in sample_not_present:
#                     y_pred.insert(int(ii)-1,0)
#             y_pred = pd.DataFrame(y_pred)
#             ranking = y_pred.rank(ascending=False,method="first")
#             ranking = np.array(ranking)
#             pred_ranking=[]
#             for ij in ranking:
#                 for jj in ij:
#                     pred_ranking.append(jj)
#             y_test=list(y_test)
#             if len(y_test)!=21:
#                 for ij in sample_not_present:
#                     y_test.insert(int(ij)-1,0)
#             y_test=pd.DataFrame(y_test)

#             ranking1 = y_test.rank(ascending=False,method="first")
#             ranking1 = np.array(ranking1)
#             real_ranking = []
#             for ij in ranking1:
#                 for jj in ij:
#                     real_ranking.append(jj)
#             print(real_ranking)
#             print(pred_ranking)
#             kapa_score.append(sklearn.metrics.cohen_kappa_score(np.array(real_ranking),np.array(pred_ranking)))

#             reliability_data=[list(real_ranking),list(pred_ranking)]
#             krippen_score.append(krippendorff.alpha(reliability_data=reliability_data))
#             real_5=[]
#             pred_5=[]
#             for r in range(1,6):
#                 if r in real_ranking:
#                     index=real_ranking.index(r)
#                     real_5.append(index+1)
#                 if r in pred_ranking:
#                     index1=pred_ranking.index(r)
#                     pred_5.append(index1+1)
#             count=0
#             for rk in pred_5:
#                 if rk in real_5:
#                     count=count+1
#             precisson_5.append(count/5)
#             data = {'Dataset':[i]*21,'Classifier':[j]*21,'Metric':[metric]*21,'predicted_score': list(y_pred[0]),'predicted_ranking': list(pred_ranking),'original_score': list(y_test[0]),'original_ranking':list(real_ranking)}
#             new_file=pd.DataFrame(data)
#             path="/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/exp_validation/test_file_result/"+j+'_'+metric+'_'+i
#             new_file.to_csv(path,index=False)
#             data={}


#         csvwriter.writerow([i, j, kapa_score[0], kapa_score[1], kapa_score[2], kapa_score[3], kapa_score[4]])
#         csvwriter1.writerow([i, j, krippen_score[0], krippen_score[1], krippen_score[2], krippen_score[3], krippen_score[4]])
#         csvwriter2.writerow([i, j, precisson_5[0], precisson_5[1], precisson_5[2], precisson_5[3], precisson_5[4]])
# write_file.close()
# write_file1.close()
# write_file2.close()