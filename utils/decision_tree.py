#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 21:09:05 2020

@author: kalrashid
"""



import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from dtreeplt import dtreeplt
import graphviz 


le = preprocessing.LabelEncoder()
clf = DecisionTreeClassifier(random_state=0,
                             max_depth=3)


X = Affinity[['clothes', 'kitchen', 'small_appliances', 'toys', 'house_keeping']]
y =  Affinity[['Labels']] # Target variable

# How many elements per Cluster


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=1) # 70% training and 30% tes

# Create Decision Tree classifer object
#clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X, y)

clf.feature_importances_

plot_tree(clf, filled=True)




dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 







dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=X.columns,
                                class_names = list(set(Affinity.iloc[:,-1].to_string())),
                                filled=True,
                                rounded=True,
                                special_characters=True)  
graph = graphviz.Source(dot_data)



to_class = {'clothes':[99,10,5, 0],
        'kitchen':[1, 60, 5, 90],
        'small_appliances':[0, 5, 75, 2],
        'toys':[0,5, 5, 7], 
        'house_keeping':[0, 20, 10, 1]}

# Creates pandas DataFrame. 
to_class = pd.DataFrame(to_class, index =['cust1', 'cust2', 'cust3', 'cust4']) 


# Classify these new elements
