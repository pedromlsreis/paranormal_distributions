#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 07:09:18 2019

@author: kalrashid, pedromlsreis
"""

# importing all the libraries
import pandas as pd
from pip._internal import main as pipmain
from utils.data_extraction import data_extract
from utils.preprocessing import preprocessing_df


# setting display options
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)
pd.set_option('display.float_format', '{:.2f}'.format)

# source: https://docs.python.org/3/library/sqlite3.html


def run(path=str, profile_after_extract=False, nb_exploration=False):
    # data extraction
    _, df = data_extract(path)
    # exploring the data
    if profile_after_extract:
        try:
            import pandas_profiling
        except ImportError as e:
            print(e.args)
            pipmain(['install', 'pandas_profiling'])
            import pandas_profiling
        prof = df.profile_report(style={'full_width': True}, title='Pandas Profiling Report')
        prof.to_file(output_file="./out/df_profiling.html")

    # data preprocessing
    df, df_Norm = preprocessing_df(df)

    print(df.head(2))
    
    if nb_exploration:
        return df

my_path = r'./data/insurance.db'

if __name__ == "__main__":
    run(path=my_path)
