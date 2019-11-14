# Data preprocessing functions
import numpy as np


def cleaning_dataframe(df):
    df.loc[df["Birthday"] < 1900, "Birthday"] = np.nan  # turning impossible values into NaN
    df.loc[df["First_Policy"] > 2020, "First_Policy"] = np.nan
    df["Education"] = df["Education"].str.extract(r"(\d)").astype(np.float)  # turning Education into numeric
    return df


def preprocessing_dataframe(df):
    df = cleaning_dataframe(df)
    dups_df = df[df.duplicated(keep="first")].copy() # duplicated rows (showing only the duplicates)
    return df, dups_df