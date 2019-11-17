# Data preprocessing functions
import numpy as np
import pandas as pd
from scipy import stats

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



#Cleaning the data
def cleaning_dataframe(df):
    df.loc[df["Birthday"] < 1900, "Birthday"] = np.nan  # turning impossible values into NaN
    df.loc[df["First_Policy"] > 2020, "First_Policy"] = np.nan
    df["Education"] = df["Education"].str.extract(r"(\d)").astype(np.float)  # turning Education into numeric
    return df


#Creating Dummy variables for Area and Education
def adding_dummies(df, cols):
    df_with_dummies = pd.get_dummies(df, columns = cols, drop_first=True)
    return df_with_dummies

#Dealing with Missing Values




#Dealing with Outliers

#not sure too much about this.
def find_anomalies(ys, threshold = 3):
    ys_std = np.std(ys)
    ys_mean = np.mean(ys)
    anomaly_cut_off = ys_std * threshold
    
    lower_limit = ys_mean - anomaly_cut_off 
    upper_limit = ys_mean + anomaly_cut_off
    
    anomalies = [a for a in ys if a > upper_limit or a < lower_limit]

    return anomalies

#second way, but it needs a dataframe with columns that are numeric and have outliers

def fancy_anomalies(df):
    df = df[~(np.abs(df-df.mean()) > (3*df.std()))]
    return df


#Data transformation
    

def preprocessing_dataframe(df):
    df = cleaning_dataframe(df)
    dups_df = df[df.duplicated(keep="first")].copy() # duplicated rows (showing only the duplicates)
    return df, dups_df
