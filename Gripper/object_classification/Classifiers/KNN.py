
from numpy import cov
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# from Gripper.object_classification.Classifiers.import_serial import test

from matplotlib.pyplot import (figure, plot, title, xlabel, ylabel, 
                               colorbar, imshow, xticks, yticks, show)


def KNN_performance(first_attribute, metric = 'minkowski'):
    # Names of data objects
    dataobjectNames = [
        'full_bottle','empty_bottle',
        'full_can','empty_can',
        'hard_ball','tennis_ball',
        'tomato_good', 'tomato_bad'
        ]
  

    # Number of classes
    C = len(dataobjectNames)
    # Attribute values
    i = 0
    for name in dataobjectNames:
        df = pd.read_excel("Gripper/object_classification/Classifiers/Dataset.xlsx", sheet_name=name)
        locals()["df_"+name] = df.drop("Unnamed: 0", axis=1)
        X_help = np.array(locals()["df_"+name])
        if i == 0:
            X = X_help
            i = 1
        else:
            X = np.concatenate((X,X_help), axis = 0)

    y=[]
    # Class indices
    for i in range(C):
        if first_attribute in dataobjectNames[i]:
            y=np.concatenate((y, np.ones(100)*i), axis = 0)
        else:
            y=np.concatenate((y, np.ones(100)*-1), axis = 0)
            
    C = len(set(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  shuffle=True)
    

    # Plot the training data points (color-coded) and test data points.
    # figure(1)
    # styles = ['.b', '.r', '.g', '.y']
    # for c in range(C):
        # class_mask = (y_train==c)
        # plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])

    # K-nearest neighbors
    K=3

    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=2
    if metric == 'minkowski':
        metric_params = {} # no parameters needed for minkowski
    elif metric == 'cosine' :
        metric_params = {} # no parameters needed for cosine
    elif metric == 'mahalanobis':
        metric_params={'V': cov(X_train, rowvar=False)}

    # Fit classifier and classify the test points
    knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                        metric=metric,
                                        metric_params=metric_params)
    knclassifier.fit(X_train, y_train)
    y_est = knclassifier.predict(X_test)
    print(X_test)
    print(y_test)
    print(y_est)
    # # Plot the classfication results
    # styles = ['ob', 'or', 'og', 'oy']
    # for c in range(C):
    #     class_mask = (y_est==c)
    #     plot(X_test[class_mask,0], X_test[class_mask,1], styles[c], markersize=10)
    #     plot(X_test[class_mask,0], X_test[class_mask,1], 'kx', markersize=8)
    # title('Synthetic data classification - KNN');

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

def KNN(first_attribute, metric = 'minkowski'):
    # Names of data objects
    dataobjectNames = [
        'full_bottle','empty_bottle',
        'full_can','empty_can',
        'hard_ball','tennis_ball',
        'tomato_good', 'tomato_bad'
        ]
    # # Names of data objects
    # dataobjectNames = [
    #     'full_bottle','empty_bottle',
    #     'full_can','empty_can',
    #     'tomato_good', 'tomato_bad'
    #     ]
    # Number of classes
    C = len(dataobjectNames)
    # Attribute values
    i = 0
    for name in dataobjectNames:
        df = pd.read_excel("Gripper/object_classification/Classifiers/Dataset.xlsx", sheet_name=name)
        locals()["df_"+name] = df.drop("Unnamed: 0", axis=1)
        X_help = np.array(locals()["df_"+name])
        if first_attribute in name:
            if i ==0:
                X = X_help
                i = 1
            else:
                X = np.concatenate((X,X_help), axis = 0)
        else:
            n = 20  # for 2 random indices
            index = np.random.choice(X_help.shape[0], n, replace=False)  
            if i == 0:
                X = X_help[index,:]
                i = 1
            else:
                X = np.concatenate((X,X_help[index,:]), axis = 0)
        

    y=[]
    # Class indices
    for i in range(C):
        if first_attribute in dataobjectNames[i]:
            y=np.concatenate((y, np.ones(100)*i), axis = 0)
        else:
            y=np.concatenate((y, np.ones(20)*-1), axis = 0)
            
        # y=np.concatenate((y, np.ones(100)*i), axis = 0)   
    print(X.shape)
    print(y.shape)
    print(X)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  shuffle=True)
    

    # Plot the training data points (color-coded) and test data points.
    figure(1)
    styles = ['.b', '.r', '.g', '.y']
    for c in range(C):
        class_mask = (y_train==c)
        # plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])

    # K-nearest neighbors
    K=3

    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=2
    if metric == 'minkowski':
        metric_params = {} # no parameters needed for minkowski
    elif metric == 'cosine' :
        metric_params = {} # no parameters needed for cosine
    elif metric == 'mahalanobis':
        metric_params={'V': cov(X_train, rowvar=False)}

    # Fit classifier and classify the test points
    knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                        metric=metric,
                                        metric_params=metric_params)
    knclassifier.fit(X_train, y_train)
    # X_test = [[0.051, 0.000, 1.261, 0.000, 0.000, 0.000, 0.000, 0.111, 0.000, 0.961, 0.586, 0.000, 0.000, 0.000]]
    # X_test = [[0,	0,	0,	0,	0.816,	0.285,	0.177,	0,	0,	0,	0,	1.091,	0.438,	0.395]]
    x_test = test()
    y_est = knclassifier.predict([x_test])
    # print(y_est)
    if y_est[0] == -1:
        result = "Not match between the tactile measure and the visual."
    else:
        result = dataobjectNames[int(y_est[0])]
    
    return result
def KNN_test(first_attribute, metric = 'minkowski'):
    # Names of data objects
    dataobjectNames = [
        'full_bottle','empty_bottle',
        'full_can','empty_can',
        'hard_ball','tennis_ball',
        'tomato_good', 'tomato_bad'
        ]

    # Number of classes
    C = len(dataobjectNames)
    # Attribute values
    i = 0
    for name in dataobjectNames:
        df = pd.read_excel("Gripper/object_classification/Classifiers/Dataset.xlsx", sheet_name=name)
        locals()["df_"+name] = df.drop("Unnamed: 0", axis=1)
        X_help = np.array(locals()["df_"+name])
        if first_attribute in name:
            if i ==0:
                X = X_help
                i = 1
            else:
                X = np.concatenate((X,X_help), axis = 0)
        else:
            n = 25  # for 2 random indices
            index = np.random.choice(X_help.shape[0], n, replace=False)  
            if i == 0:
                X = X_help[index,:]
                i = 1
            else:
                X = np.concatenate((X,X_help[index,:]), axis = 0)


    y=[]
    j = 1
    # Class indices
    for i in range(C):
        if first_attribute in dataobjectNames[i]:
            y=np.concatenate((y, np.ones(100)*i), axis = 0)
        else:
            y=np.concatenate((y, np.ones(25)*-1), axis = 0)
            # continue
            
        # y=np.concatenate((y, np.ones(100)*i), axis = 0)   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  shuffle=True)
    


    # Plot the training data points (color-coded) and test data points.
    figure(1)
    styles = ['.b', '.r', '.g', '.y']
    for c in range(C):
        class_mask = (y_train==c)
        # plot(X_train[class_mask,0], X_train[class_mask,1], styles[c])

    # K-nearest neighbors
    K=3

    # Distance metric (corresponds to 2nd norm, euclidean distance).
    # You can set dist=1 to obtain manhattan distance (cityblock distance).
    dist=2
    if metric == 'minkowski':
        metric_params = {} # no parameters needed for minkowski
    elif metric == 'cosine' :
        metric_params = {} # no parameters needed for cosine
    elif metric == 'mahalanobis':
        metric_params={'V': cov(X_train, rowvar=False)}

    # Fit classifier and classify the test points
    knclassifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                        metric=metric,
                                        metric_params=metric_params)
    knclassifier.fit(X_train, y_train)
    # X_test = [[0.051, 0.000, 1.261, 0.000, 0.000, 0.000, 0.000, 0.111, 0.000, 0.961, 0.586, 0.000, 0.000, 0.000]]
    # X_test = [[0,	0,	0,	0,	0.816,	0.285,	0.177,	0,	0,	0,	0,	1.091,	0.438,	0.395]]
    
    # Number of classes
    C = len(dataobjectNames)
    # Attribute values
    i = 0
    # dataobjectNames = ["test_full_bottle", "test_empty_bottle", "test_full_can_vertical", "test_empty_can", "test_hard_ball", "test_tennis_ball"] 

    # for name in dataobjectNames:
    #     df = pd.read_excel("/home/belencastellote/Documents/OCVT_fixed/Gripper/object_classification/Classifiers/Dataset_test.xlsx", sheet_name=name)
    #     locals()["df_"+name] = df.drop("Unnamed: 0", axis=1)
    #     X_help = np.array(locals()["df_"+name])

    #     if i ==0:
    #         X_test = X_help
    #         i = 1
    #     else:
    #         X_test = np.concatenate((X_test,X_help), axis = 0)
        # if first_attribute in name:
        #     if i ==0:
        #         X_test = X_help
        #         i = 1
        #     else:
        #         X_test = np.concatenate((X_test,X_help), axis = 0)
        # else:
        #     continue
        #     if i == 0:
        #         X = X_help[:25,:]
        #         i = 1
        #     else:
        #         X = np.concatenate((X,X_help[:25,:]), axis = 0)

    # y_test=[]
    # # Class indices
    # for i in range(C):
    #     if first_attribute in dataobjectNames[i]:
    #         y_test=np.concatenate((y_test, np.ones(10)*i), axis = 0)
    #     else:
    #         y_test=np.concatenate((y_test, np.ones(10)*-1), axis = 0)
    y_est = knclassifier.predict(X_test)

    # if y_est[0] == -1:
    #     result = "Not match between the tactile measure and the visual."
    # else:
    #     result = dataobjectNames[int(y_est[0])]
    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_est);
    accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
    print("Accuracy: ", accuracy)
    hard_true, hard_false, soft_true, soft_false, som_else_true, som_else_false=0,0,0,0,0,0
    if first_attribute=="bottle":
        for i in range(len(y_test)):
            if y_test[i] == 0:
                if y_est[i] == 0:
                    hard_true+=1
                else:
                    hard_false+=1
            elif y_test[i] == 1:
                if y_est[i] == 1:
                    soft_true+=1
                else:
                    soft_false+=1

        print("Accuracy on hard: ", float(hard_true)/float(hard_false+hard_true))
        print("Accuracy on soft: ", float(soft_true)/float(soft_false+soft_true))
    elif first_attribute=="can":
        for i in range(len(y_test)):
            if y_test[i] == 2:
                if y_est[i] == 2:
                    hard_true+=1
                else:
                    hard_false+=1
            elif y_test[i] == 3:
                if y_est[i] == 3:
                    soft_true+=1
                else:
                    soft_false+=1
        print("Accuracy on hard: ", float(hard_true)/float(hard_false+hard_true))
        print("Accuracy on soft: ", float(soft_true)/float(soft_false+soft_true))
    if first_attribute=="ball":
        for i in range(len(y_test)):
            if y_test[i] == 4:
                if y_est[i] == 4:
                    hard_true+=1
                else:
                    hard_false+=1
            elif y_test[i] == 5:
                if y_est[i] == 5:
                    soft_true+=1
                else:
                    soft_false+=1
        print("Accuracy on hard: ", float(hard_true)/float(hard_false+hard_true))
        print("Accuracy on soft: ", float(soft_true)/float(soft_false+soft_true))
    if first_attribute=="tomato":
        for i in range(len(y_test)):
            if y_test[i] == 6:
                if y_est[i] == 6:
                    hard_true+=1
                else:
                    hard_false+=1
            elif y_test[i] == 7:
                if y_est[i] == 7:
                    soft_true+=1
                else:
                    soft_false+=1
        print("Accuracy on hard: ", float(hard_true)/float(hard_false+hard_true))
        print("Accuracy on soft: ", float(soft_true)/float(soft_false+soft_true))
    for i in range(len(y_test)):
            if y_test[i] == -1:
                if y_est[i] == -1:
                    som_else_true+=1
                else:
                    som_else_false+=1
    print("Accuracy on something else: ", float(som_else_true)/float(som_else_false+som_else_true))
    # return result    
# I should add a class on random objects.
# print("-----------------------------------------BOTTLE-----------------------------------------")
# KNN_test("bottle", metric = 'minkowski')
# print("-----------------------------------------CAN-----------------------------------------")
# KNN_test("can", metric = 'minkowski')
# print("-----------------------------------------BALL-----------------------------------------")
# KNN_test("ball", metric = 'minkowski')
print("-----------------------------------------TOMATO-----------------------------------------")
KNN_test("tomato", metric = 'minkowski')
print("-----------------------------------------END-----------------------------------------")