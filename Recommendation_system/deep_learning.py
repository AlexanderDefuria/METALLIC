
import numpy as np
import pandas as pd
import data_handling
import distance
import hypersphere
import overlapping,kmeans
from scipy.stats import rankdata
from keras.models import Sequential
from keras.layers import Dense
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
# df = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Metafeature/features.csv") #features_regression.csv
df = pd.read_csv("C:/Ronald/uOttawa/6900/Metallic-main/Metafeature/features.csv") #features_regression.csv
# filename="/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Dataset/BreastCancer.csv" #input("enter the file name:")
filename="C:/Ronald/uOttawa/6900/Metallic-main/Dataset/BreastCancer.csv" #input("enter the file name:")
metrics='F1'#input("enter the metrics to be considered:")
print("select the Classifiers:")
print("1.KNN")
print("2.DT")
print("3.GNB")
print("4.SVM")
print("5.RF")
classifier="RF"#input("enter the classifier:")
rows=df[df[classifier]==1]
y_train=np.array(rows[metrics]) #y_train
x_train=rows.iloc[:,1:49] #Xtrain ready
X, y = data_handling.loading(filename)
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
Silhouettescore, DaviesBouldinscore, Calinskiharabazscore, Cohesionscore, Separationscore, RMSSTDscore, RSscore, XBscore, Adjustedrandomscore, Adjusted_mutual_info_score, Fowlkes_mallows_score, Normalizedmutualinfoscore = kmeans.k_Means(X, y)
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
# model = xg.XGBRegressor(colsample_bytree=0.4,gamma=0,learning_rate=0.07,max_depth=3,min_child_weight=1.5,n_estimators=10000,reg_alpha=0.75,reg_lambda=0.45,subsample=0.6,seed=42)
model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
# compile the keras model
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['MeanSquaredError'])
# fit the keras model on the dataset
# model.fit(X, y, epochs=150, batch_size=10)
x_train=np.array(x_train)
model.fit(np.array(x_train),y_train, epochs = 10)

print(model.summary())
maxLen = max(map(len, test_x))
[row.extend([0]*(maxLen - len(row))) for row in test_x]
test_x = np.array(test_x, dtype=np.float32)
y_pred=model.predict(test_x)
#ranking=rankdata(y_pred,)
y_pred=pd.DataFrame(y_pred)
#print('y_pred:',y_pred)
ranking=y_pred.rank(ascending=False)
ranking=np.array(ranking)
list_rank=[]
for i in ranking:
    for j in i:
        list_rank.append(float(j))
sort_ranking=list_rank.copy()
sort_ranking.sort()
top_3=sort_ranking[:3]
index_list=[]
for i in top_3:
    index_list.append(list_rank.index(i))
#print(index_list)
first=sample(index_list[0]+1)
second=sample(index_list[1]+1)
third=sample(index_list[2]+1)
print("The recommended sampling methods are:")
print("1.",first)
print("2.",second)
print("3.",third)


#new_rows=np.array(new_rows)