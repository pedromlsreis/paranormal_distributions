#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 05:38:25 2020

@author: kalrashid
"""
#import required modules
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

os.listdir()
os.getcwd()
os.chdir('/home/kalrashid/Dropbox/nova/data_mining/project/')







from utils.data_extraction import data_extract
from utils.preprocessing import preprocessing_df

from utils.preprocessing import standardize_data

#housekeeping
# setting display options
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)
pd.set_option('display.float_format', '{:.2f}'.format)


#creating pandas dataframe from the sql datafile

my_path = r'./data/insurance.db' #path of the data file
_, df = data_extract(my_path)

#sanity check
df.head()


#preprocess the dat
df= preprocessing_df(df)

#getting two dataframe df<- original after imputation, df_norm <- is normalised all the continous values.


df.head()

#checking to ensure df_norm has any nan values
df.isna().any()



"""
#####################################
######### Agglomerative #############
#####################################
"""
#lets check whether 

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from pylab import rcParams

# The final result will use the sklearn

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm





df.columns
InsValue = df.loc[:,['CMV', 'Health',
               'Life',
               'Work_Compensation',
               'Household']].reindex()

# Consumption Normalize

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Ins_Norm = scaler.fit_transform(InsValue)

Ins_Norm = pd.DataFrame(Ins_Norm, columns = InsValue.columns)


# We need scipy to plot the dendrogram 


plt.figure(figsize=(10,5))
plt.style.use('seaborn-whitegrid')

#Scipy generate dendrograms

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
Z = linkage(Ins_Norm,
            method = 'ward')#method='single', 'complete', 'ward'

#https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
dendrogram(Z,
           truncate_mode='lastp',
           p= 10,
           orientation = 'top',
           leaf_rotation=90,
           leaf_font_size=10,
           show_contracted=True,
           show_leaf_counts=True, color_threshold=50, above_threshold_color='k')



plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster Size')
plt.ylabel('Distance')
#plt.axhline(y=50)
plt.show()



k = 4

Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')

#Replace the test with proper data
my_HC = Hclustering.fit(Ins_Norm)

my_labels = pd.DataFrame(my_HC.labels_)
my_labels.columns =  ['Labels']


# Do the necessary transformations
columns = Ins_Norm.columns

Affinity = pd.DataFrame(pd.concat([pd.DataFrame(Ins_Norm), my_labels], axis=1),
                        columns = [*Ins_Norm.columns, 'Labels'])

to_revert = Affinity.groupby(['Labels'])[columns].mean()

hier_result = pd.DataFrame(scaler.inverse_transform(X=to_revert),
                            columns = columns)



"""
#####################################
############# K-means ###############
#####################################
"""
#importing necessary libraries

#kmeans with one hot encoding


from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

#slicing df to exclude categorical data

cat_vars = ['Area', 'Education', 'Children']

#create one hot encoding for categorical variables
one_hot_encoded_education = pd.get_dummies(df['Education'], prefix = 'Education')    
one_hot_encoded_area = pd.get_dummies(df['Area'], prefix = 'Area')
one_hot_encoded_children = pd.get_dummies(df['Children'], prefix = 'Children')

df_hot = pd.concat([one_hot_encoded_area, one_hot_encoded_children, one_hot_encoded_education], axis = 1)

len(df_hot)
df_hot.index = np.arange(0, len(df_hot))
len(df_hot)

df_hot.isna().any()
#seperating continous columns from categorical
temp_df = df.drop(cat_vars, axis=1)
len(temp_df)

temp_df.head()

temp_df.reset_index(drop=True, inplace=True)
df_hot.reset_index(drop=True, inplace=True)

df_Joined = pd.concat([temp_df, df_hot], axis = 1)


keamnsHotdf_Norm = scaler.fit_transform(df_Joined)
keamnsHotdf_Norm = pd.DataFrame(keamnsHotdf_Norm, columns = df_Joined.columns)

keamnsHotdf_Norm.isna().any()



Sum_of_squared_distances = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(keamnsHotdf_Norm)
    Sum_of_squared_distances.append(km.inertia_)

# Plot the elbow
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


n_clusters = 5

kmeans = KMeans(n_clusters=n_clusters, 
                random_state=0,
                n_init = 10,
                max_iter = 2000).fit(keamnsHotdf_Norm)

kmeans_clusters = pd.DataFrame(kmeans.cluster_centers_)

my_clusters = kmeans.cluster_centers_

scaler.inverse_transform(X = my_clusters)


kmeansHot_clusters = pd.DataFrame(scaler.inverse_transform(X = my_clusters),
                           columns = df_Joined.columns)

kmeansHot_unique, kmeansHot_counts = np.unique(kmeans.labels_, return_counts = True)

kmeansHot_counts

#modified code from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

silhouette_avg = silhouette_score(keamnsHotdf_Norm, kmeans.labels_)

print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(keamnsHotdf_Norm, kmeans.labels_)


cluster_labels = kmeans.labels_

import matplotlib.cm as cm
y_lower = 100

fig = plt.figure()
ax = fig.add_subplot(111)


#ax.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
ax.set_ylim([0, keamnsHotdf_Norm.shape[0] + (n_clusters + 1) * 10])

for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.rainbow(float(i) / n_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color,
                      edgecolor=color, 
                      alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 100  # 10 for the 0 samples
    
    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    
plt.show()



"""
#####################################
############# K-means ###############
#####################################
"""
#importing necessary libraries


#seperating continous columns from categorical
kmeans_df = df.drop(cat_vars, axis=1)

kmeans_df.head()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
kmeansDf_Norm = scaler.fit_transform(kmeans_df)
kmeansDf_Norm = pd.DataFrame(kmeansDf_Norm, columns = kmeans_df.columns)

kmeansDf_Norm.isna().any()

kmeansDf_Norm.head()

Sum_of_squared_distances = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(kmeansDf_Norm)
    Sum_of_squared_distances.append(km.inertia_)

# Plot the elbow
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, 
                random_state=0,
                n_init = 10,
                max_iter = 2000).fit(kmeansDf_Norm)

temp_clusters = pd.DataFrame(kmeans.cluster_centers_)

my_clusters = kmeans.cluster_centers_

scaler.inverse_transform(X = my_clusters)



kmeans_clusters = pd.DataFrame(scaler.inverse_transform(X = my_clusters),
                           columns = kmeansDf_Norm.columns)




#modified code from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

silhouette_avg = silhouette_score(kmeansDf_Norm, kmeans.labels_)

print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(kmeansDf_Norm, kmeans.labels_)


cluster_labels = kmeans.labels_

import matplotlib.cm as cm
y_lower = 100

fig = plt.figure()
ax = fig.add_subplot(111)


#ax.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
ax.set_ylim([0, df_Norm.shape[0] + (n_clusters + 1) * 10])

for i in range(n_clusters):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.rainbow(float(i) / n_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color,
                      edgecolor=color, 
                      alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 100  # 10 for the 0 samples
    
    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    
plt.show()


"""
#####################################
############# K-modes ###############
#####################################
"""
#importns
from kmodes.kmodes import KModes

#need to define df_cat with categorical values with no standarisation


MIN = df['CMV'].min()
Q1 = df['CMV'].quantile(0.25)
Median = df['CMV'].median()
Q3 = df['CMV'].quantile(0.75)
MAX = df['CMV'].max()


CMV_cat = pd.cut(df['CMV'], bins = [MIN, Q1, Median, Q3, MAX], labels = ['least_valued', 'low_valued', 'valued', 'highly_valued'])

vars_for_keamns = ['Area', 'Education', 'Children', 'CMV']

VE_Cat = df[vars_for_keamns].copy().astype('str')
VE_Cat.insert(4, 'CMV Cats', CMV_cat)
VE_Cat.drop(['CMV'], axis=1, inplace = True)

VE_Cat['CMV Cats'].isna().sum()
VE_Cat.dropna(inplace=True)
VE_Cat.isna().any()


km = KModes(n_clusters = 4, init = 'random', n_init = 50, verbose=1)

kmode_clusters = km.fit_predict(VE_Cat)

cat_centroids = pd.DataFrame(km.cluster_centroids_,
                             columns = vars_for_keamns)

unique, counts = np.unique(km.labels_, return_counts = True)

cat_counts = pd.DataFrame(np.asarray((unique, counts)).T, columns = ['Label', 'Number'])

cat_centroids = pd.concat([cat_centroids, cat_counts], axis = 1)




"""
#####################################
############### SOM #################
#####################################
"""
import joblib
import random

SOMDf_Norm = scaler.fit_transform(kmeans_df)
SOMDf_Norm = pd.DataFrame(kmeansDf_Norm, columns = kmeans_df.columns)

SOMDf_Norm.isna().any()

SOMDf_Norm.head()

from sompy.sompy import SOMFactory
from sompy.visualization.plot_tools import plot_hex_map
import logging

df_Norm.head()
temp_df = df[['Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Salary',
       'CMV', 'Customer_Years']]


X = temp_df.values
names = temp_df.columns


sm = SOMFactory().build(data = X,
               mapsize=(10,10),
               normalization = 'var',
               initialization='random', #'pca'
               component_names=names,
               lattice= 'hexa',
               training = 'seq')#'seq','batch'

sm.train(n_job=4,
         verbose='info',
         train_rough_len=30,
         train_finetune_len=100)


som_clusters = pd.DataFrame(sm._data, columns = names)

my_labels = pd.DataFrame(sm._bmu[0])
    
som_clusters = pd.concat([som_clusters,my_labels], axis = 1)

som_clusters.columns = [*names, 'Labels']



from sompy.visualization.mapview import View2DPacked
view2D  = View2DPacked(10,10,"", text_size=7)
view2D.show(sm, col_sz=5, what = 'codebook',)#which_dim="all", denormalize=True)
plt.show()



from sompy.visualization.mapview import View2D
view2D  = View2D(10,10,"", text_size=7)
view2D.show(sm, col_sz=5, what = 'codebook',)#which_dim="all", denormalize=True)
plt.show()


from sompy.visualization.bmuhits import BmuHitsView
vhts  = BmuHitsView(12,12,"Hits Map",text_size=7)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=10, cmap="autumn", logaritmic=False)



# K-Means Clustering
from sompy.visualization.hitmap import HitMapView
sm.cluster(3)
hits  = HitMapView(10,10,'Clustering', text_size=7)
a=hits.show(sm, labelsize=12)



