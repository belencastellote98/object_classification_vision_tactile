# exercise 6.3.1

from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from numpy import cov
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load Matlab data file and extract variables of interest
# mat_data = loadmat('../Data/synth1.mat') # <-- change the number to change dataset

# Names of data objects
dataobjectNames = [
    'full_bottle',
    'empty_bottle'
    ]
# Number of classes:
N=2

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  shuffle=True)


# X = mat_data['X']
# X_train = mat_data['X_train']
# X_test = mat_data['X_test']
# y = mat_data['y'].squeeze()
# y_train = mat_data['y_train'].squeeze()
# y_test = mat_data['y_test'].squeeze()
# attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = ["0", "1", "half"]
N, M = X.shape
C = len(classNames)


# Plot the training data points (color-coded) and test data points.
figure(1)
styles = ['.b', '.r', '.g', '.y']
for c in range(C):
    class_mask = (y_train==c)
    plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])


# K-nearest neighbors
K=3

# Distance metric (corresponds to 2nd norm, euclidean distance).
# You can set dist=1 to obtain manhattan distance (cityblock distance).
dist=2
metric = 'minkowski'
metric_params = {} # no parameters needed for minkowski

# You can set the metric argument to 'cosine' to determine the cosine distance
#metric = 'cosine' 
#metric_params = {} # no parameters needed for cosine

# To use a mahalonobis distance, we need to input the covariance matrix, too:
#metric='mahalanobis'
#metric_params={'V': cov(X_train, rowvar=False)}

# Fit classifier and classify the test points
knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                    metric=metric,
                                    metric_params=metric_params)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)
print(X_test)
print(y_test)
print(y_est)
# Plot the classfication results
styles = ['ob', 'or', 'og', 'oy']
for c in range(C):
    class_mask = (y_est==c)
    plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
title('Synthetic data classification - KNN');

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_est);
accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
figure(2);
imshow(cm, cmap='binary', interpolation='None');
colorbar()
xticks(range(C)); yticks(range(C));
xlabel('Predicted class'); ylabel('Actual class');
title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate));

show()

