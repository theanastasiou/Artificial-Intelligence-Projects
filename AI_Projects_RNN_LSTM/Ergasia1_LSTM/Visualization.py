import pandas as pd
import glob
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import csv
import matplotlib.dates as mdates
import seaborn as sns

path = r'/home/at/Desktop/Texniti_2/' # use your path'

filename = path + "3MonthGrouping.csv"
df = pd.read_csv(filename , header=0)

df['DateTime'] =pd.to_datetime(df['DateTime'])
# # print("WTF")
df.set_index(pd.DatetimeIndex(df['DateTime']),inplace=True)


sns.boxplot(data=df, x='Day', y='Global_active_power')
plt.show()

cols2 = ['Sub_metering_1','Sub_metering_2','Sub_metering_3']
df_monthly_mean = df[cols2].resample('M').mean()
#plot data
fig, ax = plt.subplots(figsize=(15,7))
#set ticks every week
for nm in cols2:
    ax.plot(df_monthly_mean[nm],label=nm)
    ax.set_ylim(0, 6000)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    #set major ticks format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.show()



cols = ['Voltage','Global_active_power','Global_reactive_power','Sub_metering_1','Sub_metering_2','Sub_metering_3']
# The min_periods=360 argument accounts for a few isolated missing days in the
# wind and solar production time series
cols1 = ['Sub_metering_1','Sub_metering_2','Sub_metering_3']
df_365d = df[cols1].rolling(window=365, center=True, min_periods=360).mean()
# Plot 365-day rolling mean time series of wind and solar power
fig, ax = plt.subplots()
for nm in cols1:
    ax.plot(df_365d[nm], label=nm)
    # Set x-ticks to yearly interval, adjust y-axis limits, add legend and labels
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_ylim(0, 6000)
    ax.legend()
    ax.set_ylabel('Production (GWh)')
    ax.set_title('Trends in Electricity Production (365-d Rolling Means)');
plt.show()

# Specify the data columns we want to include (i.e. exclude Year, Month, Weekday Name)
# Resample to weekly frequency, aggregating with mean
df_weekly_mean = df[cols].resample('W').mean()

# Start and end of the date range to extract
start, end = '2009-01', '2009-06'
# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()
ax.plot(df.loc[start:end, 'Sub_metering_1'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(df_weekly_mean.loc[start:end, 'Sub_metering_1'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('Sub_metering_1')
ax.legend()
plt.show()

# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()
ax.plot(df.loc[start:end, 'Sub_metering_2'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(df_weekly_mean.loc[start:end, 'Sub_metering_2'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('Sub_metering_2')
ax.legend()
plt.show()

# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()
ax.plot(df.loc[start:end, 'Sub_metering_3'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(df_weekly_mean.loc[start:end, 'Sub_metering_3'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('Sub_metering_3')
ax.legend()
plt.show()

# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()
ax.plot(df.loc[start:end, 'Global_active_power'],
marker='.', linestyle='-', linewidth=0.5, label='Daily')
ax.plot(df_weekly_mean.loc[start:end, 'Global_active_power'],
marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample')
ax.set_ylabel('Global_active_power')
ax.legend()
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['Global_active_power','Global_reactive_power','Sub_metering_1','Sub_metering_2','Sub_metering_3'], axes):
    sns.boxplot(data=df, x='Day', y=name, ax=ax)
    ax.set_ylabel('GWh')
    ax.set_title(name)
    # Remove the automatic x-axis label from all but the bottom subplot
    if ax != axes[-1]:
        ax.set_xlabel('')
plt.show()


sns.boxplot(data=df, x='Month', y='Sub_metering_1')
plt.show()


ax = df.loc['2008', 'Sub_metering_1'].plot()
ax.set_ylabel('Sub_Metering_1_2008')
plt.show()

ax = df.loc['2008', 'Sub_metering_2'].plot()
ax.set_ylabel('Sub_Metering_2_2008')
plt.show()

ax = df.loc['2008', 'Sub_metering_3'].plot()
ax.set_ylabel('Sub_Metering_3_2008')
plt.show()

ax = df.loc['2007-01':'2008-01', 'Sub_metering_1'].plot(marker='o', linestyle='-')
ax.set_ylabel('Sub_Metering_1_07_08')
plt.show()

# df['Month'] = df.index.month #find week based on statetime
# df.set_index(df['Month'], inplace=True)
# pp = sns.pairplot(df[cols], size=1.8, aspect=1.8,
#                   plot_kws=dict(edgecolor="k", linewidth=0.5),
#                   diag_kind="kde", diag_kws=dict(shade=True))

# fig = pp.fig 
# fig.subplots_adjust(top=0.93, wspace=0.3)
# t = fig.suptitle('Dataset Attributes Pairwise Plots', fontsize=14)  
# plt.show()  

# df.set_index(pd.DatetimeIndex(df['DateTime']),inplace=True)

# Correlation Matrix Heatmap
f, ax = plt.subplots(figsize=(10, 6))
corr = df.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Dataset Attributes Correlation Heatmap', fontsize=14)
plt.show()


sns.set(rc={'figure.figsize':(11, 4)})
df['Sub_metering_1'].plot(linewidth=0.5)
plt.show()


sns.set(rc={'figure.figsize':(11, 4)})
df[cols].plot(linewidth=0.5)
plt.show()


# pp = sns.pairplot(df[cols], size=1.8, aspect=1.8,
#                   plot_kws=dict(edgecolor="k", linewidth=0.5),
#                   diag_kind="kde", diag_kws=dict(shade=True))
# plt.show()  




cols_plot = ['Sub_metering_1','Sub_metering_2']
axes =df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_xlabel('Month')
    ax.set_ylabel('Daily Totals (GWh)')

plt.show()

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for name, ax in zip(['Sub_metering_1','Sub_metering_2','Sub_metering_3'], axes):
    sns.boxplot(data=df, x='Quarter', y=name, ax=ax)
    ax.set_ylabel('GWh')
    ax.set_title(name)
    # Remove the automatic x-axis label from all but the bottom subplot
    if ax != axes[-1]:
        ax.set_xlabel('')
plt.show()

pd.pivot_table(df, values = "Global_active_power", 
               columns = "Year", index = "Month").plot(subplots = True, figsize=(12, 12), layout=(3, 5), sharey=True);
plt.show()

fig, ax = plt.subplots()
ax.plot(df['Month'], color='black', label='Month')
df[['Sub_metering_1','Sub_metering_2']].plot.area(ax=ax, linewidth=0)
ax.legend()
ax.set_ylabel('Monthly Total (GWh)')
plt.show()





# df = pd.read_csv(filename , header=0,nrows = 730)

# df['DateTime'] =pd.to_datetime(df['DateTime'])
# # # print("WTF")
# df.set_index(pd.DatetimeIndex(df['DateTime']),inplace=True)

# import matplotlib.dates as mdates

# fig, ax1 = plt.subplots(figsize=(10, 5))
# ax1.set(xlabel='', ylabel='Total # of trips started')
# ax1.plot(df["DateTime"], df.Global_active_power, color='g')
# ax1.plot(df["DateTime"], df.Global_active_power, color='b')

# ax1.xaxis.set(
#     major_locator=mdates.DayLocator(),
#     major_formatter=mdates.DateFormatter("\n\n%A"),
#     minor_locator=mdates.HourLocator((0, 12)),
#     minor_formatter=mdates.DateFormatter("%H"),
# )
# plt.show()
