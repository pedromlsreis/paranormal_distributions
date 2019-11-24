#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 07:09:18 2019

@author: kalrashid, pedromlsreis
"""

# importing all the libraries
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pip._internal import main as pipmain

try:
    import pandas_profiling
except:
    pipmain(['install', 'pandas_profiling'])
    import pandas_profiling

from utils.data_extraction import data_extract
from utils.preprocessing import preprocessing_dataframe, adding_dummies, fancy_anomalies, remove_outlier


# setting display options
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)
pd.set_option('display.float_format', '{:.2f}'.format)

# source: https://docs.python.org/3/library/sqlite3.html

my_path = r'.\data\insurance.db'

def run(path = str, profile = bool):
    # data extraction
    original_df, df = data_extract(path)

    #exploring the data
    if profile:
        profile = df.profile_report(style={'full_width':True}, title='Pandas Profiling Report')
        profile.to_file(output_file="./out/df_profiling.html")

    # data preprocessing
    df, dups_df = preprocessing_dataframe(df)
    #adding dummy variables
    df = adding_dummies(df, cols = ['Area', 'Education'])

    #removing outliers
    #testing to remove outliers using z score. But some of the results are fucking waaaaaack!

    #columns of which outliers need to be identified
    col_names = ['Motor','Household','Health','Life','Work_Compensation']

    df_1 = df.copy()
    # df_1.loc[col_names] = remove_outlier(df, col_names)

    print(df.head(2))
    # print(df_1.head(2))


if __name__ == "__main__":
    run(path = my_path, profile = False)
