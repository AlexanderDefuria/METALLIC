import sklearn
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
labels = ['dog', 'cat', 'pig']
y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'dog'])
y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes= labels, y=y_true)
prfs = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels)
recalls = prfs[1] #Specificity in Binary Classification
print(weights)
print(recalls)
s = sum(weights)
new_wei = weights/s
print(new_wei)
req=new_wei*recalls
print(req)
                

# data = pd.read_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Dataset/abalone9-18.csv")
# final_column = data[data.columns[-1]]