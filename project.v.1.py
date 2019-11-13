#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 07:09:18 2019

@author: kalrashid
"""

import sqlite3
import os
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib
from matplotlib import pyplot as plt

import pandas_profiling

#setting display
pd.set_option('display.width', 4000)
pd.set_option('max_colwidth', 4000)
pd.set_option('max_rows', 100)
pd.set_option('max_columns', 200)
pd.set_option('display.float_format', '{:.2f}'.format)

# source: https://docs.python.org/3/library/sqlite3.html

# my_path = '/home/kalrashid/Dropbox/nova/data_mining/project/data/insurance.db'
my_path = r'C:\Users\pedro\OneDrive\Documents\MAA\Data_Mining\paranormal_distributions\data\insurance.db'

# connect to the database
conn = sqlite3.connect(my_path)

query = """
    select
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
	from
    Engage as e
	join LOB as l on l."Customer Identity" = e."Customer Identity"
    order by
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







