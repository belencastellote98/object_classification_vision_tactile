import numpy as np
import pandas as pd

from sklearn import tree
# from cleaning_sampling import cleaning_function
from import_serial import test

import matplotlib.pyplot as plt
from matplotlib.image import imread

# Names of data objects
dataobjectNames = [
    'full_bottle',
    'empty_bottle'
    ]

# Number of classes:


# Attribute names
attributeNames = [
    'T1',
    'T2',
    'T3',
    'T4',
    'T5',
    'T6',
    'T7',
    'T8',
    'T9',
    'T10',
    'T11',
    'T12',
    'T13',
    'T14'    
    ]

# Attribute values
i=0
for name in dataobjectNames:
    df = pd.read_excel("/home/belencastellote/Documents/OCVT_fixed/Gripper/object_classification/Classifiers/Dataset.xlsx", sheet_name=name)
    locals()["df_"+name] = df.drop("Unnamed: 0", axis=1)
    X_help = np.array(locals()["df_"+name])
    if i==0:
        X=X_help
        i=1
    else:
        X = np.concatenate((X,X_help), axis = 0)

y=[]
N=2
flag = 0
# Class indices
for i in range(N):
    if flag == 0:
        y=np.ones(100)*i
        flag = 1
    else:
        y = np.concatenate((y, np.ones(100)*i), axis = 0)

# Class names
classNames = ['hard', 'soft']
    
# Number data objects, attributes, and classes
N, M = X.shape
print(N)
C = len(classNames)

# Entropy regression tree
criterion='entropy'
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=5)
dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=1.0/N)
dtc = dtc.fit(X,y)

x = test()
print(x)
x_class = dtc.predict([x])
print(x_class)
print("The object is identified as: " + dataobjectNames[x_class[0].astype(int)])