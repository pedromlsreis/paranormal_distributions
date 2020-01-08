#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 19:15:22 2020

@author: kalrashid
"""

import sqlite3
import pandas as pd


def data_extract(path_to_db):
    # connect to the database
    conn = sqlite3.connect(path_to_db)

    query = """
        SELECT
        e."Customer Identity",
        "First Policy´s Year",
        "Brithday Year",
        "Educational Degree",
        "Gross Monthly Salary",
        "Geographic Living Area",
        "Has Children (Y=1)",
        "Customer Monetary Value",
        "Claims Rate",
        l."Premiums in LOB: Motor",
        l."Premiums in LOB: Household",
        l."Premiums in LOB: Health",
        l."Premiums in LOB:  Life",
        l."Premiums in LOB: Work Compensations"
        FROM
        Engage AS e
        JOIN LOB AS l ON l."Customer Identity" = e."Customer Identity"
        ORDER BY
        e."Customer Identity";
    """

    data_df = pd.read_sql_query(query, conn)
    df = data_df.copy()  # let's keep a copy of the original data

    # remaining column names to manageable variable names
    column_names = ['ID', 'First_Policy', 'Birthday', 'Education',
                    'Salary', 'Area', 'Children', 'CMV', 'Claims',
                    'Motor', 'Household', 'Health', 'Life',
                    'Work_Compensation']
    # renaming the columns
    df.columns = column_names
    # seting 'ID' as index
    df.set_index('ID', inplace=True, drop=True)
    return data_df, df


my_path = r'/home/kalrashid/Dropbox/nova/data_mining/project/data/insurance.db'
_, df = data_extract(my_path)


"""new file"""



import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sb
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

"""
Steps to follow, according to the lectures:
Data preparation
 Exploratory data analysis
 Detecting outliers
 Dealing with missing values
 Data discretization
 Imbalanced learning and data generation

Data preprocessing
 The curse of dimensionality
 Identifying informative attributes/features
 Creating attributes/features
 Dimensionality reduction
  Relevancy
  Redundancy
 Data standardization

"""

# undocumented handy function: df._get_numeric_data()


def cleaning_df(df):
    # removing duplicate rows
    df = df[~df.duplicated(keep="last")]
    # turning impossible values into NaN
    df.loc[df["Birthday"] < 1900, "Birthday"] = np.nan
    df.loc[df["Birthday"] > 2016, "Birthday"] = np.nan
    df.loc[df["First_Policy"] > 2016, "First_Policy"] = np.nan
    df.loc[df["Birthday"] > df["First_Policy"], "First_Policy"] = np.nan

    # turning Education into numeric
    df["Education"] = df["Education"].str.extract(r"(\d)").astype(np.float)
    return df


def add_dummies(df, cols):
    """Adds dummy columns to selected variables using the One Hot Encoding method.
    Drops the first column."""
    df_with_dummies = pd.get_dummies(df, columns=cols, drop_first=True)
    return df_with_dummies


def outlier_conditions(df):
    """
    Sets the condition for the identification of outliers in a dataframe
    """
    # ~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR)))


    # Q1 = df['col'].quantile(.25)
    # Q3 = df['col'].quantile(.75)
    # mask = d['col'].between(q1, q2, inclusive=True)
    # iqr = d.loc[mask, 'col']



    return ~(np.abs(df - df.mean()) > (3 * df.std()))


def remove_outliers(df, cols):
    """
    Replaces outliers by NaNs.
    Selected columns must be numerical.
    """
    outlier_df_cond = outlier_conditions(df)
    outliers_count = (
        (df[cols] == df[outlier_df_cond][cols]) == False
        )[cols].sum()
    
    temp_df = df[cols].copy()
    outlier_tempdf_cond = outlier_conditions(temp_df)
    temp_df = temp_df[outlier_tempdf_cond]
    
    df.loc[:, cols] = temp_df.loc[:, cols].copy()
    return df, outliers_count


# def remove_outliers(df, cols):
#     """
#     Replaces outliers by NaNs.
#     Selected columns must be numerical.
#     """
#     ~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))


#     for col in cols:
#         Q1 = df[col].quantile(.25)
#         Q3 = df[col].quantile(.75)

#         mask = df[col].between(Q1, Q3, inclusive=True)
#         IQR = df.loc[mask, col]
        
#         df.loc[
#             (df[col] < (Q1 - 1.5 * IQR)) | ( )
#             , col
#         ] = np.nan

#         cond = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))



#     return df, outliers_count


def handle_nans(df, cols):
    """
    Replaces NaNs by column mean.
    Selected columns must be continuous.
    """
    df.fillna(df.mean()[cols], inplace=True)
    return df


def handle_premium_nans(df, cols):
    """
    Replaces NaNs with 0.
    Selected columns must be continuous.
    """
    for col in cols:
        df[col].fillna(0, inplace=True)
    return df


def handle_cat_nans(df, cols):
    """
    Uses a Random Forest classifier to predict and impute the nan values 
    for each categorical column given in `cols`.
    """
    # Xcols = []

    # for cat_col in cols:
    #     if df[cat_col].isna().any().sum() != 0:
    #         Xcols.append(cat_col)
    
    # if len(Xcols) != 0:
    #     for nan_col in Xcols:
    #         X_train = df.loc[:, df.columns.difference( list(set(Xcols) - set(list(nan_col))) )].values
    #         y_train = df.loc[:, nan_col].values
    #         clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=2019)
    #         clf.fit(X_train, y_train)
    #         X_test = df.loc[df[cat_col].isna(), Xcols].copy()
    #         y_pred = clf.predict(X_test)
            
    #         for pred, index in zip(y_pred, X_test.index.tolist()):
    #             df.loc[index, cat_col] = pred

    #         print(f'NaN values of "{cat_col}" column were imputed.')
    #     return df
    # else:
    #     return df
    for col in cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def standardize_data(df, cols):
    """Standardizes data from `cols`.
    cols -> list
    """
    df_Norm = df[cols].copy()
    df_Norm[cols] = StandardScaler().fit_transform(df[cols])
    return df, df_Norm


def feature_selection(df):
    corr = df.corr(method='pearson')

    # Obtain Correlation and plot it
    plt.figure(figsize=(16,6))

    h_map = sb.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap='PRGn', annot=True, linewidths=.5)

    bottom, top = h_map.get_ylim()
    h_map.set_ylim(bottom + 0.5, top - 0.5)

    plt.show()


def feature_eng(df):
    """
    Creates useful features from the original ones in the dataframe.
    """
    if "Birthday" in df.columns:
        df["Age"] = 2016 - df["Birthday"]
        del df["Birthday"]
    
    if "First_Policy" in df.columns:
        df["Customer_Years"] = 2016 - df["First_Policy"]
        del df["First_Policy"]
    
    return df


def dim_reduction(df):
    """
    Applies Principal Component Analysis (PCA) to the dataframe.
    """
    x = df.values
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal_comp_1', 'principal_comp_2'])
    print(pca.explained_variance_ratio_)
    return principalDf


def preprocessing_df(df):
    #separation of variables
    ValueEngage = ['Age', 'Education', 'Salary', 'Area', 'Children', 'CMV', 'Customer_Years']

    ConsAff = ['Motor', 'Household', 'Health', 'Life', 'Work_Compensation']
    Cat_Values = ["Area", "Education", "Children"]
    
    collist = []
    collist.extend(ConsAff)
    collist.extend(Cat_Values)

    df = cleaning_df(df)
    df, outliers_count = remove_outliers(df, df.columns)
    
    df = handle_nans(df, df.columns.difference(collist))
    df = handle_premium_nans(df, ConsAff)
    df = handle_cat_nans(df, Cat_Values)

    df.loc[:, ["First_Policy", "Birthday", "Salary"]] = df[["First_Policy", "Birthday", "Salary"]].round().astype(np.int32)

    df.loc[:, Cat_Values] = df[Cat_Values].astype("category")
    
    df = feature_eng(df)
    df, df_Norm = standardize_data(df, [*ConsAff, 'Salary', 'CMV', 'Customer_Years'])

    # df = dim_reduction(df)
    
    return df, df_Norm



df, df_Norm = preprocessing_df(df)

df_Norm['Area'], df_Norm['Education'], df_Norm['Children'] = df['Area'], df['Education'], df['Children']

df_Norm.isna().any()


# Cluster analysis

"""
1. Define variables to use
2. Define similarity/dissimilarity criterion between entities
3. Define a clustering algorithm to create groups of similar entities
4. Analyse it and validate the resulting solution.
"""

import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

"""
#####################################
######### Agglomerative #############
#####################################
"""
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#temp_df = scaler.fit_transform(df_Norm)
temp_df = df_Norm[['Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Salary',
       'CMV', 'Customer_Years']]
df_Norm.columns

plt.figure(figsize=(10, 7))
plt.title("Customer Dendograms")
dend = shc.dendrogram(shc.linkage(temp_df, method='ward'))



n_clusters = 4

cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
cluster.fit_predict(df_Norm)

plt.figure(figsize=(10, 7))
plt.scatter(df_Norm.iloc[:,0], df_Norm.iloc[:,1], c=cluster.labels_, cmap='rainbow')

plt.show()

# Do the necessary transformations

from sklearn.cluster import AgglomerativeClustering

scaler = StandardScaler()

dfvals = scaler.fit_transform(df_Norm)
k = 4

Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')
my_HC = Hclustering.fit(dfvals)

aff = df_Norm.copy()
aff["Labels"] = pd.DataFrame(my_HC.labels_)

to_revert = aff.groupby(['Labels']).mean()
final_result = pd.DataFrame(scaler.inverse_transform(X=to_revert),
                            columns = aff.columns.difference(["Labels"]))

final_result

"""
temp_df = scaler.fit_transform(temp_df)

k = 5

Hclustering = AgglomerativeClustering(n_clusters=k,
                                      affinity='euclidean',
                                      linkage='ward')
my_HC = Hclustering.fit(temp_df)

my_labels = pd.DataFrame(my_HC.labels_)
my_labels.columns =  ['Labels']

aff = pd.DataFrame(pd.concat([pd.DataFrame(temp_df), my_labels], axis=1),
                        columns = ['Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Salary',
       'CMV', 'Customer_Years','Labels'])

to_revert = aff.groupby(['Labels'])['Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Salary',
       'CMV', 'Customer_Years'].mean()

final_result = pd.DataFrame(scaler.inverse_transform(X=to_revert),
                            columns = ['Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Salary',
       'CMV', 'Customer_Years'])


"""
"""
#####################################
############# K-means ###############
#####################################
"""
#importing necessary libraries
from sklearn.cluster import KMeans




Sum_of_squared_distances = []
K = range(1,20)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_Norm)
    Sum_of_squared_distances.append(km.inertia_)

# Plot the elbow
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters, 
                random_state=0,
                n_init = 10,
                max_iter = 2000).fit(df_Norm)

kmeans_clusters = pd.DataFrame(kmeans.cluster_centers_, columns = df_Norm.columns)



#silhouette analysis
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


#modified code from http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

def silplot(X, clusterer, pointlabels=None):
    cluster_labels = clusterer.labels_
    n_clusters = clusterer.n_clusters
    
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(11,8.5)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters = ", n_clusters,
          ", the average silhouette_score is ", silhouette_avg,".",sep="")

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(0,n_clusters+1):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], marker='.', s=200, lw=0, alpha=0.7,
                c=colors, edgecolor='k')
    xs = X.iloc[:, 0]
    ys = X.iloc[:, 1]
    
    if pointlabels is not None:
        for i in range(len(xs)):
            plt.text(xs[i],ys[i],pointlabels[i])

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % int(i), alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

silplot(df_Norm, kmeans)
plt.show()



"""
#####################################
############# K-modes ###############
#####################################
"""

#need to define df_cat with categorical values with no standarisation

from kmodes.kmodes import KModes

VE_Cat = df_Norm[['Education', 'Area', 'Children']].astype('str')


km = KModes(n_clusters = 4, init = 'random', n_init = 50, verbose=1)

kmode_clusters = km.fit_predict(VE_Cat)

cat_centroids = pd.DataFrame(km.cluster_centroids_,
                             columns = ['Education', 'Area', 'Children'])

unique, counts = np.unique(km.labels_, return_counts = True)

cat_counts = pd.DataFrame(np.asarray((unique, counts)).T, columns = ['Label', 'Number'])

cat_centroids = pd.concat([cat_centroids, cat_counts], axis = 1)



"""
#####################################
############### SOM #################
#####################################
"""

#from sklearn.externals import joblib
import joblib
import random



from sompy.sompy import SOMFactory
from sompy.visualization.plot_tools import plot_hex_map
import logging

temp_df = df_Norm[['Motor', 'Household', 'Health', 'Life', 'Work_Compensation', 'Salary',
       'CMV', 'Customer_Years']]
df_Norm.columns


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


final_clusters = pd.DataFrame(sm._data, columns = names)

my_labels = pd.DataFrame(sm._bmu[0])
    
final_clusters = pd.concat([final_clusters,my_labels], axis = 1)

final_clusters.columns = [*names, 'Labels']


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



"""
#####################################
########### Mean-Shift ##############
#####################################
"""

