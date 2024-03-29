import numpy as np
import pandas as pd

from sklearn import tree
# from cleaning_sampling import cleaning_function
# from Gripper.object_classification.Classifiers.import_serial import test

import matplotlib.pyplot as plt
from matplotlib.image import imread

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def tree_class (first_attribute):
    # Names of data objects
    dataobjectNames = [
        'full_bottle','empty_bottle',
        'full_can','empty_can',
        'hard_ball','tennis_ball',
        ]

    # dict_dataObjects = {"bottle":["full", "empty"], "can": ["full", "empty"], "ball": ["hard","tennis"]}


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
    N=len(dataobjectNames)
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

        
    # Number data objects, attributes, and classes
    N, M = X.shape
    print(N)
    
    # Entropy regression tree
    criterion='entropy'
    # dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=5)
    dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=1.0/N)
    dtc = dtc.fit(X,y)

    x = test()
    # print(x)
    y_est = dtc.predict([x])

    print(y_est)
    if y_est[0] == -1:
        result = "Not match between the tactile measure and the visual."
    else:
        result = dataobjectNames[int(y_est[0])]
    
    return result
def tree_test(first_attribute, metric = 'minkowski'):
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
    N, M = X_train.shape
    # Entropy regression tree
    criterion='entropy'
    # dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=5)
    dtc = tree.DecisionTreeClassifier(criterion=criterion, min_samples_split=1.0/N)
    dtc = dtc.fit(X_train,y_train)

    
    
    # Number of classes
    # C = len(dataobjectNames)
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
    y_est = dtc.predict(X_test)

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
# tree_test("bottle")
# print("-----------------------------------------CAN-----------------------------------------")
# tree_test("can")
# print("-----------------------------------------BALL-----------------------------------------")
# tree_test("ball")
# print("-----------------------------------------END-----------------------------------------")