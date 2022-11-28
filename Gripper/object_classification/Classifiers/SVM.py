
# Support Vector Machine
# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from Gripper.object_classification.Classifiers.import_serial import test

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

def SVM_performance(first_attribute, kernel = "linear"):
# Names of data objects
    dataobjectNames = [
        'full_bottle','empty_bottle',
        'full_can','empty_can',
        'hard_ball','tennis_ball',
        ]

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
            print("HOLAAA\n")
        else:
            y=np.concatenate((y, np.ones(100)*-1), axis = 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  shuffle=True)
    # Fitting the classifier into the Training set
    if kernel == "linear":
        svc = SVC(kernel = 'linear', random_state = 0)
    elif kernel == "quadratic":
        svc = SVC(kernel = 'poly', degree = 2, random_state = 0)
    elif kernel == "cubic":
        svc = SVC(kernel = 'poly', degree = 3, random_state = 0)
    svc.fit(X_train, y_train)
    y_pred=svc.predict(X_test)
    print('Model accuracy score with given hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
    plt.figure()
    plt.plot(y_pred)
    plt.plot(y_test)
    plt.show()
    return

def SVM(first_attribute, kernel="linear"):
    # Names of data objects
    dataobjectNames = [
        'full_bottle','empty_bottle',
        'full_can','empty_can',
        'hard_ball','tennis_ball',
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
    # Class indices
    for i in range(C):
        if first_attribute in dataobjectNames[i]:
            y=np.concatenate((y, np.ones(100)*i), axis = 0)
        else:
            y=np.concatenate((y, np.ones(25)*-1), axis = 0)
    
    # For this dataset I wont be using any PCA because is not difference


    # # Normalize data for doing PCA
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # # Doing PCA
    # pca = PCA(n_components=None)
    # X_pca = pca.fit(X)
    # X_trans = pca.transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  shuffle=True)
    # Fitting the classifier into the Training set
    if kernel == "linear":
        svc = SVC(kernel = 'linear', random_state = 0)
    elif kernel == "quadratic":
        svc = SVC(kernel = 'poly', degree = 2, random_state = 0)
    elif kernel == "cubic":
        svc = SVC(kernel = 'poly', degree = 3, random_state = 0)
    svc.fit(X_train, y_train)
    y_pred=svc.predict(X_test)
    print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    # Predicting the test set results
    x_test = test()
    y_est = svc.predict([x_test])

    if y_est[0] == -1:
        result = "Not match between the tactile measure and the visual."
    else:
        result = dataobjectNames[int(y_est[0])]
    
    return result

def SVM_test(first_attribute, kernel = "linear"):
    # Names of data objects
    dataobjectNames = [
        'full_bottle','empty_bottle',
        'full_can','empty_can',
        'hard_ball','tennis_ball',
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

    # Class indices
    for i in range(C):
        if first_attribute in dataobjectNames[i]:
            y=np.concatenate((y, np.ones(100)*i), axis = 0)
        else:
            y=np.concatenate((y, np.ones(25)*-1), axis = 0)
            
        # y=np.concatenate((y, np.ones(100)*i), axis = 0)   

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,  shuffle=True)

    # Fitting the classifier into the Training set
    if kernel == "linear":
        svc = SVC(kernel = 'linear', random_state = 0)
    elif kernel == "quadratic":
        svc = SVC(kernel = 'poly', degree = 2, random_state = 0)
    elif kernel == "cubic":
        svc = SVC(kernel = 'poly', degree = 3, random_state = 0)
    svc.fit(X_train, y_train)
    y_pred=svc.predict(X_test)
    print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    # Predicting the test set results

    y_est = svc.predict(X_test)
    # X_test = [[0.051, 0.000, 1.261, 0.000, 0.000, 0.000, 0.000, 0.111, 0.000, 0.961, 0.586, 0.000, 0.000, 0.000]]
    # X_test = [[0,	0,	0,	0,	0.816,	0.285,	0.177,	0,	0,	0,	0,	1.091,	0.438,	0.395]]
    
    # Number of classes
    C = len(dataobjectNames)
    # Attribute values
    i = 0
    # dataobjectNames = ["test_full_bottle", "test_empty_bottle", "test_full_can", "test_empty_can", "test_hard_ball", "test_tennis_ball"] 

    # for name in dataobjectNames:
    #     df = pd.read_excel("/home/belencastellote/Documents/OCVT_fixed/Gripper/object_classification/Classifiers/Dataset_test.xlsx", sheet_name=name)
    #     locals()["df_"+name] = df.drop("Unnamed: 0", axis=1)
    #     X_help = np.array(locals()["df_"+name])

    #     if i ==0:
    #         X_test = X_help
    #         i = 1
    #     else:
    #         X_test = np.concatenate((X_test,X_help), axis = 0)
    #     # if first_attribute in name:
    #     #     if i ==0:
    #     #         X_test = X_help
    #     #         i = 1
    #     #     else:
    #     #         X_test = np.concatenate((X_test,X_help), axis = 0)
    #     # else:
    #     #     continue
    #     #     if i == 0:
    #     #         X = X_help[:25,:]
    #     #         i = 1
    #     #     else:
    #     #         X = np.concatenate((X,X_help[:25,:]), axis = 0)

    # y_test=[]
    # # Class indices
    # for i in range(C):
    #     if first_attribute in dataobjectNames[i]:
    #         y_test=np.concatenate((y_test, np.ones(10)*i), axis = 0)
    #     else:
    #         y_test=np.concatenate((y_test, np.ones(10)*-1), axis = 0)
    y_est=svc.predict(X_test)
    
    # print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred))    # if y_est[0] == -1:
    #     result = "Not match between the tactile measure and the visual."
    # else:
    #     result = dataobjectNames[int(y_est[0])]
    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, y_est);
    accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
    print("Accuracy: ", accuracy)
    hard_true, hard_false, soft_true, soft_false,som_else_true, som_else_false=0,0,0,0,0,0
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
    for i in range(len(y_test)):
            if y_test[i] == -1:
                if y_est[i] == -1:
                    som_else_true+=1
                else:
                    som_else_false+=1
    print("Accuracy on something else: ", float(som_else_true)/float(som_else_false+som_else_true))
    # return result    
# # I should add a class on random objects.
# print("-----------------------------------------BOTTLE-----------------------------------------")
# SVM_test("bottle", kernel = "quadratic")
# print("-----------------------------------------CAN-----------------------------------------")
# SVM_test("can", kernel = "quadratic")
# print("-----------------------------------------BALL-----------------------------------------")
# SVM_test("ball", kernel = "quadratic")
# print("-----------------------------------------END-----------------------------------------")
# SVM_performance("bottle")
