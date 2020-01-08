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
    # Q1 = df['col'].quantile(.25)
    # Q3 = df['col'].quantile(.75)
    # mask = d['col'].between(q1, q2, inclusive=True)
    # iqr = d.loc[mask, 'col']
    # ~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR)))

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
    df_Norm['Area'], df_Norm['Education'], df_Norm['Children'] = df['Area'], df['Education'], df['Children']

    return df, df_Norm
