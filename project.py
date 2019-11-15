#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 07:09:18 2019

@author: kalrashid
"""

# importing all the libraries
import sqlite3
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
from utils.preprocessing import preprocessing_dataframe


#setting display
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)
pd.set_option('display.float_format', '{:.2f}'.format)

# source: https://docs.python.org/3/library/sqlite3.html

my_path = '/home/kalrashid/Dropbox/nova/data_mining/project/data/insurance.db'
#my_path = r'C:\Users\pedro\OneDrive\Documents\MAA\Data_Mining\paranormal_distributions\data\insurance.db'

# connect to the database
conn = sqlite3.connect(my_path)

query = """
    SELECT
	e."Customer Identity",
	"First Policy´s Year",
	"Brithday Year",
	"Educational Degree",
	"Gross Monthly Salary",
	"Geographic Living Area",
	"Has Children (Y=1)",
	"Customer Monetary Value",
	"Claims Rate",
    l."Premiums in LOB: Motor",
	l."Premiums in LOB: Household",
	l."Premiums in LOB: Health",
	l."Premiums in LOB:  Life",
	l."Premiums in LOB: Work Compensations"
	FROM
    Engage AS e
	JOIN LOB AS l ON l."Customer Identity" = e."Customer Identity"
    ORDER BY
	e."Customer Identity";
"""

data_df = pd.read_sql_query(query, conn)

df = data_df.copy()  # let's keep a copy of the original data

#remaining column names to manageable variable names

column_names = ['ID', 'First_Policy', 'Birthday', 'Education', 'Salary', 'Area', 'Children', 'CMV',
                'Claims', 'Motor', 'Household', 'Health', 'Life', 'Work_Compensation']


#renaming the columns
df.columns = column_names

#seting 'ID' as index
df.set_index('ID', inplace = True, drop = True)

#exploring the data
df.describe()
profile = df.profile_report(style={'full_width':True}, title='Pandas Profiling Report')
profile.to_file(output_file="df_profiling.html")

#data preprocessing
df, dups_df = preprocessing_dataframe(df)
#adding dummy variables
df = creating_dummies(df, cols = ['Area', 'Education'])



