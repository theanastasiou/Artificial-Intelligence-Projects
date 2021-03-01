import pandas as pd
import glob
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import csv

import seaborn as sns

#---------------------PRE PROCESING-----------------------
path = r'/home/at/Desktop/Texniti_2/' # use your path'

filename = path + "household_power_consumption.csv"
df = pd.read_csv(filename ,  delimiter=';', low_memory=False)
print(df.isna().sum())

# filename = path + "Grouping.csv"
# df = pd.read_csv(filename , header=0)


df['DateTime'] = df['Date']+ " " + df['Time']
# print(df['DateTime'][0])

df['DateTime'] =pd.to_datetime(df['DateTime'])
# print("WTF")
df.set_index(pd.DatetimeIndex(df['DateTime']),inplace=True)


df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df = df.dropna(subset=['Global_active_power'])



df['Global_active_power'].replace('?',0,inplace=True) 
df['Global_reactive_power'].replace('?',0,inplace=True) 
df['Voltage'].replace('?',0,inplace=True) 
df['Global_intensity'].replace('?',0,inplace=True) 
df['Sub_metering_1'].replace('?',0,inplace=True) 
df['Sub_metering_2'].replace('?',0,inplace=True) 
df['Sub_metering_3'].replace('?',0,inplace=True) 


df['Global_active_power'] = df['Global_active_power'].astype(float)
df['Global_reactive_power'] = df['Global_reactive_power'].astype(float)
df['Voltage'] = df['Voltage'].astype(float)
df['Global_intensity'] = df['Global_intensity'].astype(float)
df['Sub_metering_1'] = df['Sub_metering_1'].astype(float)
df['Sub_metering_2'] = df['Sub_metering_2'].astype(float)
df['Sub_metering_3'] = df['Sub_metering_3'].astype(float)
#3MonthGroupin3MonthGroupin

print("OKKK")

df = df.groupby(pd.Grouper(freq='12H'))['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3'].sum()

print("whhatt")

df['Global_active_power'].fillna(0.0, inplace=True)
df['Global_reactive_power'].fillna(0.0, inplace=True)
df['Voltage'].fillna(0.0, inplace=True)
df['Global_intensity'].fillna(0.0, inplace=True)
df['Sub_metering_1'].fillna(0.0, inplace=True)
df['Sub_metering_2'].fillna(0.0, inplace=True)
df["Sub_metering_3"].fillna(0.0, inplace=True)

# print(df.isna().sum())

# df.set_index(df['Date'], inplace=True)

df['Month'] = df.index.month #find week based on datetime
df['Quarter'] = df.index.quarter #find week based on datetime
df['Year'] = df.index.year #find week based on datetime
df['Day'] = df.index.weekday #find week based on datetime
df['Quarter'] =df.index.quarter #find quarter based on datetime

# print(df.isna().sum())
# columns=['DateTime','Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3','Year','Quarter','Day','Month']
# df = df.reindex(columns=columns)
# df = df[['DateTime','Global_active_power','Year','Quarter','Month','Day']]
# print(df)
df.to_csv("3MonthGrouping.csv")