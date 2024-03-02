import imblearn
import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
def evaluation(y_pred,y_true,state,lr_probs,X_test):
    # print("This is y true", y_true)
    # print("This is y pred", y_pred)
    au_y_true = y_true
    au_lr = lr_probs
    if state == 'multiclass':
        value='macro'
    else:
        value = 'binary'
    f1_score = sklearn.metrics.f1_score(y_true, y_pred,average=value)
    accuracy=sklearn.metrics.accuracy_score(y_true, y_pred)
    if state == 'multiclass':
        value='multiclass'
    else:
        value = 'binary'
    geometric_mean=imblearn.metrics.geometric_mean_score(y_true, y_pred,average=value) #geometric mean
    if state == 'multiclass':
        value='weighted'
    else:
        value = 'binary'
    
    precision=sklearn.metrics.precision_score(y_true, y_pred, average=value,labels=np.unique(y_pred))
    
    recall=sklearn.metrics.recall_score(y_true, y_pred,average=value)
    
    y_true = np.argmax(X_test, axis=1)
    if state =='multiclass':
        roc_auc = -1
        
    else:
        try:
            roc_auc = roc_auc_score(au_y_true, au_lr)
        except:
            roc_auc = 0.5
        
        

    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)

    if state =='multiclass':
        pr_auc = -1
        
    else:
        # print(au_y_true)
        try:
            precision, recall, _ = precision_recall_curve(au_y_true, au_lr)
            pr_auc = auc(recall, precision)
        except:
            pr_auc = 0.5

    

    weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(y_true), y=y_true)
    prfs = precision_recall_fscore_support(y_true, y_pred, average=None, labels=np.unique(y_true))
    recalls = prfs[1] #Specificity in Binary Classification
    # print(weights)
    # print(recalls)
    s = sum(weights)
    new_wei = weights/s
    # print(new_wei)
    cwa=sum(new_wei*recalls)
    # print(req)
    
    return f1_score,accuracy,geometric_mean,precision,recall,roc_auc, pr_auc, balanced_accuracy,cwa





