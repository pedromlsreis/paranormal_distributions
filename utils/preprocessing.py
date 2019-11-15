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
def creating_dummies(df, cols):
    Dummy_Vars = pd.get_dummies(df, columns = cols, drop_first=True)
    df = pd.concat([df, Dummy_Vars], axis=1)
    return df

#Dealing with Missing Values
#Dealing with Outliers
#Data transformation
    

def preprocessing_dataframe(df):
    df = cleaning_dataframe(df)
    dups_df = df[df.duplicated(keep="first")].copy() # duplicated rows (showing only the duplicates)
    return df, dups_df
