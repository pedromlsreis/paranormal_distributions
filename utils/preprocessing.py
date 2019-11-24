import pandas as pd
import numpy as np
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
    """Adds dummy columns to selected variables using the One Hot Encoding method. Drops the first column."""
    df_with_dummies = pd.get_dummies(df, columns=cols, drop_first=True)
    return df_with_dummies


#Dealing with Missing Values



#Dealing with Outliers


#another way, but it needs a dataframe with columns that are numeric and have outliers

def remove_outlier(df, cols):
    """Removes outliers and replace them by NaNs.
    Selected columns must be numerical."""
    outlier_count = ((df[cols] == df[~(np.abs(df - df.mean()) > (3 * df.std()))][cols]) == False)[cols].sum()
    temp_df = df[cols].copy()
    df = temp_df[~(np.abs(temp_df - temp_df.mean()) > (3 * temp_df.std()))].copy()
    return df, outlier_count


#Data transformation
def preprocessing_dataframe(df):
    df = cleaning_dataframe(df)
    dups_df = df[df.duplicated(keep="first")].copy() # duplicated rows (showing only the duplicates)
    return df, dups_df
