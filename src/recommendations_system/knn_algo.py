import sys
import xgboost as xg
import numpy as np
import pandas as pd
import data_handling
import src.distance as distance
import hypersphere
import overlapping,kmeans
from scipy.stats import rankdata
from math import sqrt
# sys.path.insert(1, "/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/creating_metafeatures")
sys.path.insert(1, "C:/Ronald/uOttawa/CSI 6900/Metallic-main/creating_metafeatures")
import missing_values
def knn_pred_ranking(algo, metric, dataset_name, full_data):

    def euclidean_distance(row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return sqrt(distance)

    def get_neighbors(train, test_row, num_neighbors):
        distances = list()
        c =0 
        metric_list = []
        for train_row in train:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist, c))
            c = c+1
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
            metric_list.append(distances[i][2])
        return neighbors,metric_list

    def sample(i):
        switcher = {
            1: 'None',
            2: 'SMOTE',
            3: 'NearMiss',
            4: 'SMOTEENN',
            5: 'Randomoversampling',
            6: 'ADASYN',
            7: 'BorderlineSMOTE',
            8: 'SVMSMOTE',
            9: 'RandomUnderSampler',
            10: 'ClusterCentroids',
            11: 'NearMissversion1',
            12: 'NearMissversion2',
            13: 'NearMissversion3',
            14: 'TomekLinks',
            15: 'EditedNearestNeighbours',
            16: 'RepeatedEditedNearestNeighbours',
            17: 'AllKNN',
            18: 'CondensedNearestNeighbour',
            19: 'NeighbourhoodCleaningRule',
            20: 'InstanceHardnessThreshold',
            21: 'SMOTETomek'
        }
        return switcher.get(i, "Invalid sampling method")
    df = full_data #features_regression.csv
    # filename="/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Dataset/"+dataset_name #input("enter the file name:")
    filename="C:/Ronald/uOttawa/CSI 6900/Metallic-main/Dataset/"+dataset_name+".csv" #input("enter the file name:")
    metrics=metric #input("enter the metrics to be considered:")
    # print("select the Classifiers:")
    # print("1.KNN")
    # print("2.DT")
    # print("3.GNB")
    # print("4.SVM")
    # print("5.RF")
    classifier=algo#input("enter the classifier:")
    rows=df[df[classifier]==1]
    y_train=np.array(rows[metrics]) #y_train
    x_train=rows.iloc[:,1:49] #Xtrain ready
    X, y = data_handling.loading(filename)
    X= missing_values.handle_missing_values(X,y,1)
    no_of_rows_original = X.shape[0]
    no_of_columns_original = X.shape[1]
    no_of_class=len(np.unique(y))
    if no_of_class > 2:
        state = 'multiclass'
        state_value = 1
    else:
        state = 'binaryclass'
        state_value = 0
    # unsupervised_kmeans
    Silhouettescore, DaviesBouldinscore, Calinskiharabazscore, Cohesionscore, Separationscore, RMSSTDscore, RSscore, XBscore, Adjustedrandomscore, Adjusted_mutual_info_score, Fowlkes_mallows_score, Normalizedmutualinfoscore = kmeans.kmeans_metadata(X, y)
    # imbalanced ratio
    y = y.astype(int)
    classes_data = list(np.unique(y))
    list_of_instance = [sum(y == c) for c in classes_data]
    min_instance = min(list_of_instance)
    max_instance = max(list_of_instance)
    imbalanced_ratio_before_sampling = min_instance / max_instance
    # number of hyperspheres of minority and majority and avg class
    minority_class = list_of_instance.index(min(list_of_instance))
    majority_class = list_of_instance.index(max(list_of_instance))
    hyperCentres = np.array(hypersphere.create_hypersphere(X, y))
    distance_between_classes = distance.distance(hyperCentres, np.unique(y))
    classes_hypersphere = list(set(hyperCentres[:, hyperCentres.shape[1] - 1]))
    groupsPerClass = [sum(hyperCentres[:, hyperCentres.shape[1] - 1] == c) for c in classes_hypersphere]
    minority_index = groupsPerClass.index(min(groupsPerClass))
    majority_index = groupsPerClass.index(max(groupsPerClass))
    minority_class_index = classes_hypersphere.index(minority_class)
    majority_class_index = classes_hypersphere.index(majority_class)
    minority_hypersphere = [minority_class, groupsPerClass[minority_class_index]]  # groupsPerClass[minority_class_index] will give you minority class hypersphere
    majority_hypersphere = [majority_class, groupsPerClass[majority_class_index]]
    total_hypersphere = sum(groupsPerClass)  # total number of hyperspheres
    # samples per hypershere
    total_number_instances = sum(list_of_instance)
    samplesperhypersphere = total_number_instances / total_hypersphere
    total_instance_minority = list_of_instance[minority_class]
    total_instance_majority = list_of_instance[majority_class]
    hypersphere_minority = groupsPerClass[minority_class_index]
    hypersphere_majority = groupsPerClass[majority_class_index]
    samplesperhypersphere_minority = total_instance_minority / hypersphere_minority
    samplesperhypersphere_majority = total_instance_majority / hypersphere_majority

    # volume of overlap
    over=overlapping.volume_overlap(X,y)
    test_x=[]
    test_row=[no_of_rows_original,no_of_columns_original,state_value,Silhouettescore,DaviesBouldinscore,Calinskiharabazscore,Cohesionscore,Separationscore,RMSSTDscore,RSscore,XBscore,Adjustedrandomscore,Adjusted_mutual_info_score,Fowlkes_mallows_score,Normalizedmutualinfoscore,imbalanced_ratio_before_sampling,total_hypersphere,hypersphere_minority,hypersphere_majority,samplesperhypersphere,samplesperhypersphere_minority,samplesperhypersphere_majority,distance_between_classes,over]
    for i in ['None', 'SMOTE', 'NearMiss', 'SMOTEENN', 'Randomoversampling', 'ADASYN', 'BorderlineSMOTE', 'SVMSMOTE',
            'RandomUnderSampler', 'ClusterCentroids', 'NearMissversion1', 'NearMissversion2', 'NearMissversion3',
            'TomekLinks', 'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours', 'AllKNN',
            'CondensedNearestNeighbour', 'NeighbourhoodCleaningRule', 'InstanceHardnessThreshold', 'SMOTETomek']:
        None_value = 0
        SMOTE_value = 0
        NearMiss_value = 0
        SMOTEENN_value = 0
        Randomoversampling_value = 0
        ADASYN_value = 0
        BorderlineSMOTE_value = 0
        SVMSMOTE_value = 0
        RandomUnderSampler_value = 0
        ClusterCentroids_value = 0
        NearMissversion1_value = 0
        NearMissversion2_value = 0
        NearMissversion3_value = 0
        TomekLinks_value = 0
        EditedNearestNeighbours_value = 0
        RepeatedEditedNearestNeighbours_value = 0
        AllKNN_value = 0
        CondensedNearestNeighbour_value = 0
        OneSidedSelection_value = 0
        NeighbourhoodCleaningRule_value = 0
        InstanceHardnessThreshold_value = 0
        SMOTETomek_value = 0
        if i == 'None':
            None_value = 1
        elif i == 'SMOTE':
            try:
                SMOTE_value = 1
            except:
                continue
        elif i == 'NearMiss':
            try:
                NearMiss_value = 1
            except:
                continue
        elif i == 'SMOTEENN':
            try:
                SMOTEENN_value = 1
            except:
                continue
        elif i == 'Randomoversampling':
            try:
                Randomoversampling_value = 1
            except:
                continue
        elif i == 'ADASYN':
            try:
                ADASYN_value = 1
            except:
                continue
        elif i == 'BorderlineSMOTE':
            try:
                BorderlineSMOTE_value = 1
            except:
                continue
        elif i == 'SVMSMOTE':
            try:
                SVMSMOTE_value = 1
            except:
                continue
        elif i == 'RandomUnderSampler':
            try:
                RandomUnderSampler_value = 1
            except:
                continue
        elif i == 'ClusterCentroids':
            try:
                ClusterCentroids_value = 1
            except:
                continue
        elif i == 'NearMissversion1':
            try:
                NearMissversion1_value = 1
            except:
                continue
        elif i == 'NearMissversion2':
            try:
                NearMissversion2_value = 1
            except:
                continue
        elif i == 'NearMissversion3':
            try:
                NearMissversion3_value = 1
            except:
                continue
        elif i == 'TomekLinks':
            try:
                TomekLinks_value = 1
            except:
                continue
        elif i == 'EditedNearestNeighbours':
            try:
                EditedNearestNeighbours_value = 1
            except:
                continue
        elif i == 'RepeatedEditedNearestNeighbours':
            try:
                RepeatedEditedNearestNeighbours_value = 1
            except:
                continue
        elif i == 'AllKNN':
            try:
                AllKNN_value = 1
            except:
                continue
        elif i == 'CondensedNearestNeighbour':
            try:
                CondensedNearestNeighbour_value = 1
            except:
                continue
        elif i == 'NeighbourhoodCleaningRule':
            try:
                NeighbourhoodCleaningRule_value = 1
            except:
                continue
        elif i == 'InstanceHardnessThreshold':
            try:
                InstanceHardnessThreshold_value = 1
            except:
                continue
        elif i == 'SMOTETomek':
            try:
                SMOTETomek_value = 1
            except:
                continue

        test_row.extend((None_value,SMOTE_value,NearMiss_value,SMOTEENN_value,Randomoversampling_value,ADASYN_value,BorderlineSMOTE_value,SVMSMOTE_value,RandomUnderSampler_value,ClusterCentroids_value,NearMissversion1_value,NearMissversion2_value,NearMissversion3_value,TomekLinks_value,EditedNearestNeighbours_value,RepeatedEditedNearestNeighbours_value,AllKNN_value,CondensedNearestNeighbour_value,NeighbourhoodCleaningRule_value,InstanceHardnessThreshold_value,SMOTETomek_value))
        test_x.append(test_row)
        test_row=test_row[:27]

    x_train.drop_duplicates(subset='Original Rows',inplace=True)
    req_train = x_train.iloc[:, 0:27]
    req_train = req_train.values.tolist()
    # print(req_train.head(5))
    neighbors, metric = get_neighbors(req_train, test_row, 4)
    neighbors = neighbors[1:]
    metric = metric[1:]
    # print(neighbors)
    # print(metric)
    df_num = 0
    df_dict = {}
    for i in neighbors:
        df_num = df_num + 1
        A = i[4]
        B = i[3]
        C = i[9]
        D = i[10]
        E = i[13]
        df_dict[df_num] = df.loc[(df['DaviesBouldinscore'] == A) & (df['Silhouettescore'] == B) & (df['RSscore'] ==C) & (df['XBscore'] == D) & (df['Fowlkes_mallows_score'] == E)]
    for name, df in df_dict.items():
        df1 = df_dict.get(1)
        df2 = df_dict.get(2)
        df3 = df_dict.get(3)
    imb_strategies = ['None', 'SMOTE', 'NearMiss', 'SMOTEENN', 'Randomoversampling', 'BorderlineSMOTE', 'SVMSMOTE',
            'RandomUnderSampler', 'ClusterCentroids', 'NearMissversion1', 'NearMissversion2', 'NearMissversion3',
            'TomekLinks', 'EditedNearestNeighbours', 'RepeatedEditedNearestNeighbours', 'AllKNN',
            'CondensedNearestNeighbour', 'NeighbourhoodCleaningRule', 'InstanceHardnessThreshold', 'SMOTETomek']
    value_list1 = []
    value_list2 = []
    value_list3 = []
    for i in imb_strategies:
        new_df1 = df1.loc[df1[i] == 1]
        new_df1 = new_df1.loc[new_df1[classifier] == 1, metrics]
        try:
            val = new_df1.iloc[0]
        except:
            val = 0
        value_list1.append(val)

        new_df2 = df2.loc[df2[i] == 1]
        new_df2 = new_df2.loc[new_df2[classifier] == 1, metrics]
        try:
            val2 = new_df2.iloc[0]
        except:
            val2 = 0
        value_list2.append(val2)

        new_df3 = df3.loc[df3[i] == 1]
        new_df3 = new_df3.loc[new_df3[classifier] == 1, metrics]
        try:
            val3 = new_df3.iloc[0]
        except:
            val3 = 0
        value_list3.append(val3)
        # print(new_df1)
    new_val_list = []
    for i in range(len(value_list1)):
        tot_val = value_list1[i] + value_list2[i] + value_list3[i]
        new_val_list.append(tot_val)


    req = sorted(zip(new_val_list, imb_strategies), reverse=True)
    # print(req)
    index_list=[]
    for val, imb in req:
        index_list.append(imb)
    return index_list
    # print(index_list)
    # first=index_list[0]
    # second=index_list[1]
    # third=index_list[2]
    # print("The recommended sampling methods are:")
    # print("1.",first)
    # print("2.",second)
    # print("3.",third)


