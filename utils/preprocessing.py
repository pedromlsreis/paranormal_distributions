import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    # turning impossible values into NaN
    df.loc[df["Birthday"] < 1900, "Birthday"] = np.nan
    df.loc[df["First_Policy"] > 2020, "First_Policy"] = np.nan
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


def handle_nans(df, cols):
    """
    Replaces NaNs by column mean.
    Selected columns must be numerical.
    """
    df.fillna(df.mean()[cols], inplace=True)
    return df


def standardize_data(df, cols):
    """Standardizes data from `cols`.
    cols -> list
    """
    df[cols] = StandardScaler().fit_transform(df[cols])
    return df


def preprocessing_df(df):
    df = cleaning_df(df)
    df, outliers_count = remove_outliers(df, ['Motor', 'Household', 'Health', 'Life', 'Work_Compensation'])
    df = handle_nans(df, ["Salary", "First_Policy", "Birthday", "Children", 'Motor', 'Household', 'Health', 'Life', 'Work_Compensation'])
    df = standardize_data(df, ['Motor', 'Household', 'Health', 'Life', 'Work_Compensation'])
    df = add_dummies(df, ['Area', 'Education'])
    # duplicated rows (showing only the duplicates)
    # dups_df = df[df.duplicated(keep="first")].copy()
    return df, outliers_count
