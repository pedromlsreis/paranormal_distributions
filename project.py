#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 07:09:18 2019

@author: kalrashid, pedromlsreis
"""

# importing all the libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
from utils.preprocessing import preprocessing_dataframe
from utils.data_extraction import data_extract


# setting display options
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)
pd.set_option('display.float_format', '{:.2f}'.format)

# source: https://docs.python.org/3/library/sqlite3.html

my_path = '/home/kalrashid/Dropbox/nova/data_mining/project/data/insurance.db'
#my_path = r'C:\Users\pedro\OneDrive\Documents\MAA\Data_Mining\paranormal_distributions\data\insurance.db'

# data extraction
original_df, df = data_extract(my_path)

#exploring the data
profile = df.profile_report(style={'full_width':True}, title='Pandas Profiling Report')
profile.to_file(output_file="df_profiling.html")

# data preprocessing
df, dups_df = preprocessing_dataframe(df)
#adding dummy variables
df = creating_dummies(df, cols = ['Area', 'Education'])



