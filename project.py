#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 07:09:18 2019

@author: kalrashid, pedromlsreis
"""

# importing all the libraries
import pandas as pd
from pip._internal import main as pipmain

try:
    import pandas_profiling
except ImportError as e:
    print(e.args)
    pipmain(['install', 'pandas_profiling'])
    import pandas_profiling

from utils.data_extraction import data_extract
from utils.preprocessing import preprocessing_df, add_dummies, remove_outliers


# setting display options
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)
pd.set_option('display.float_format', '{:.2f}'.format)

# source: https://docs.python.org/3/library/sqlite3.html

my_path = r'.\data\insurance.db'


def run(path=str, profile=bool):
    # data extraction
    _, df = data_extract(path)
    # exploring the data
    if profile:
        prof = df.profile_report(
            style={'full_width': True},
            title='Pandas Profiling Report'
            )
        prof.to_file(output_file="./out/df_profiling.html")

    # data preprocessing
    (df, dups_df) = preprocessing_df(df)

    # adding dummy variables
    df = add_dummies(df, cols=['Area', 'Education'])

    # removing outliers
    # testing to remove outliers using z score.
    # But some of the results are fucking waaaaaack!
    (df, outliers_count) = remove_outliers(
        df,
        cols=['Motor', 'Household', 'Health', 'Life', 'Work_Compensation']
        )

    print(f"outlier_count:\n{outliers_count}\n")
    print(df.head(2))


if __name__ == "__main__":
    run(path=my_path, profile=False)
