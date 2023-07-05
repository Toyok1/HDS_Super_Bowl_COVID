import pandas as pd
from datetime import date, timedelta
import piecewise_regression
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
import numpy as np
import scipy.stats as stats
import sdt.changepoint as c
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

file_path = "./covid_confirmed_usafacts.csv"
# Read the CSV file using pandas
dataframe = pd.read_csv(file_path)
# get rid of all the rows with county name = Statewide Unallocated
dataframe = dataframe[dataframe['County Name'] != 'Statewide Unallocated']
# get rid of all the rows with county name = Out of Country
dataframe = dataframe[dataframe['County Name'] != 'Out of Country']

# CREATION OF THE DATAFRAME
# Specify the file path or URL of the CSV file

# restrict the date range to 2021-01-23 to 2021-02-23 but keep state, countyFIPS and StateFIPS
# aggregate by state
grouped = dataframe.groupby('State').sum()
# rename the grouped column as State
grouped = grouped.rename_axis('Date').reset_index()
partGrouped = grouped[["Date"]]
# moving average starts having values on the 24th of January
start_date = date(2021, 1, 24)
super_bowl_date = date(2021, 2, 7)
sbplus7_date = date(2021, 2, 14)
sbplus14_date = date(2021, 2, 21)

difference = super_bowl_date - start_date


end_date = date(2021, 3, 7)
final_diff = end_date - start_date
delta = end_date - start_date   # returns timedelta
listdays = []
for i in range(delta.days + 1):
    day = start_date + timedelta(days=i)
    listdays.append(day.strftime("%Y-%m-%d"))
# istdays.insert(0, 'State')
otherGrouped = grouped[listdays]
# join the two dataframes
grouped = pd.concat([partGrouped, otherGrouped], axis=1)
grouped = grouped.T
grouped.columns = grouped.iloc[0]
grouped = grouped.iloc[1:, :]
# grouped.rename(columns = {'State':'Date'}, inplace = True )
grouped.head()

dataframe_list = []
print(grouped.columns)
for col in grouped.columns:
    dataframe_list.append(grouped[[col]])

for d in dataframe_list:
    # add colun with moving average for each day
    d['MAV'] = d.rolling(window=7).mean()
    # add column with the difference between the moving average of the current day and the moving average of the previous day
    d['Diff'] = d['MAV'].diff()
    d.dropna(inplace=True)
    d.index = pd.to_datetime(d.index)

print(dataframe_list[0])

array_results = []
for df in dataframe_list:
    data = df['Diff']
    data = data.values.flatten()

    # model with pymc3
    with pm.Model() as model:

        alpha = 1 / data.mean()
        lambda_1 = pm.Exponential("lambda_1", alpha)
        lambda_2 = pm.Exponential("lambda_2", alpha)
        tau = pm.DiscreteUniform("tau", lower=0, upper=len(data) - 1)
        idx = np.arange(len(data))
        lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)
        observation = pm.Poisson("obs", lambda_, observed=data)
        step = pm.Metropolis(random_seed=2)
        trace = pm.sample(10000, tune=5000, step=step,
                          return_inferencedata=True)
        array_results.append([df.columns.values[0], trace])
        print("Results for " + df.columns.values[0] + ":")
        print(trace.posterior)
        # print(trace.values)
print(array_results)
