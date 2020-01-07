import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    Selected columns must be continuous.
    """
    df.fillna(df.mean()[cols], inplace=True)
    return df


def handle_cat_nans(df, cols):
    """
    Uses a Random Forest classifier to predict and impute the nan values 
    for each categorical column given in `cols`.
    """
    used_cat_cols = []

    for cat_col in cols:
        used_cat_cols.append(cat_col)
        aux_df = df.loc[df[cat_col].isna() == False, df.columns.difference(
            list(set(cols) - set(used_cat_cols))
        )].copy()

        aux_df[cat_col] = aux_df[cat_col].round().astype(np.int8)

        Xcols = aux_df.columns.tolist()
        Xcols.remove(cat_col)
        X_train = aux_df.loc[:, Xcols].values
        y_train = aux_df.loc[:, cat_col].values

        clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=2019)
        clf.fit(X_train, y_train)

        X_test = df.loc[df[cat_col].isna(), Xcols]    
        y_pred = clf.predict(X_test)
        
        for pred, index in zip(y_pred, X_test.index.tolist()):
            df.loc[index, cat_col] = pred
        print(f'NaN values of "{cat_col}" column were imputed.')

    return df


def standardize_data(df, cols):
    """Standardizes data from `cols`.
    cols -> list
    """
    df[cols] = StandardScaler().fit_transform(df[cols])
    return df


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
    premiums_cols = ["Motor", "Household", "Health", "Life", "Work_Compensation"]
    categorical_cols = ["Area", "Education", "Children"]
    
    df = cleaning_df(df)
    df, outliers_count = remove_outliers(df, premiums_cols)
    df = handle_nans(df, ["Salary", "First_Policy", "Birthday", "Motor", "Household", "Health", "Life", "Work_Compensation"])
    df = handle_cat_nans(df, categorical_cols)

    df[["First_Policy", "Birthday", "Salary"]] = df[["First_Policy", "Birthday", "Salary"]].round().astype(np.int32)
    df[categorical_cols] = df[categorical_cols].astype("category")
    
    df = feature_eng(df)
    df = standardize_data(df, premiums_cols)

    # df = dim_reduction(df)
    
    return df, outliers_count
