import numpy as np
import pandas as pd
import math
import glob,kmeans
from sklearn.preprocessing import LabelEncoder
# files = sorted(glob.glob("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Dataset/*.csv"))
files = sorted(glob.glob("C:/Ronald/uOttawa/CSI 6900/Metallic-main/Dataset/*.csv"))
for file_new in files:
    data = pd.read_csv(file_new)
    # new_file_name = file_new.replace("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Dataset/", "")
    new_file_name = file_new.replace("C:/Ronald/uOttawa/CSI 6900/Metallic-main/Dataset/", "")
    # new_file_name = new_file_name.replace(".csv", "")
    le = LabelEncoder()
    data.Class = le.fit_transform(data.Class)
    # data.to_csv("/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/New_dataset/"+new_file_name,index=False)
    data.to_csv("C:/Ronald/uOttawa/CSI 6900/Metallic-main/New_dataset/"+new_file_name,index=False)