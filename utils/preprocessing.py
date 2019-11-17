# Data preprocessing functions
import numpy as np
import pandas as pd




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

def outliers_(ys, threshold = 3):
    print(ys.dtypes)
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    return np.where(np.abs(z_scores) > threshold)


#Data transformation
    

def preprocessing_dataframe(df):
    df = cleaning_dataframe(df)
    dups_df = df[df.duplicated(keep="first")].copy() # duplicated rows (showing only the duplicates)
    return df, dups_df
