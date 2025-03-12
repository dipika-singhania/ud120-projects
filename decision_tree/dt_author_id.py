#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

tree_model = DecisionTreeClassifier(min_samples_split=40)
clf = tree_model.fit(features_train, labels_train)

predictions = clf.predict(features_test)

accuracy = sum(predictions == labels_test) * 100.0 / len(labels_test)

print("accuracy = ", accuracy)

#########################################################


