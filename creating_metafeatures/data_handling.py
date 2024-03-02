import numpy as np
import csv
from sklearn import datasets
import sys
# sys.path.insert(1, "/Volumes/Education-Imp/UOttawa Master's/Final_Project/METALLIC/Dataset")
sys.path.insert(1, "C:/Ronald/uOttawa/6900/Metallic-main/Dataset")
def loading(file_name):
    d, ts = csv_loading(file_name)  # d is data, ts is targets

    return d, ts




def csv_loading(csv_name):
    '''
    Loading in CSV files
    :param csv_name: name of CSV file
    :return: Numpy array of data and targets,
            data: 2 dimensional numpy array with shape (x, y)
            targets: 1 dimensional numpy array with shape (x,)
    '''
    d, ts = [], []

    with open(csv_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        row_count = 0
        for row in csv_reader:
            try:
                if row_count == 0:
                    row_count += 1
                else:
                    temp_d = []
                    for i in range(len(row)):
                        if i < len(row) - 1:
                            try:
                                temp_d.append(float(row[i]))
                            except ValueError:
                                temp_d.append(float("nan"))
                        else:
                            try:
                                ts.append(int(row[i]))
                            except ValueError:
                                ts.append(float("nan"))

                    d.append(temp_d)
            except ValueError: # Skipping the lines with N/A or any other invalid values
                continue

    return np.array(d), np.array(ts)