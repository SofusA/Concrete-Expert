""".

Gotta catch em all

"""
import os
# import xlrd
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
from scipy.linalg import svd
from plotly import offline
import plotly.plotly as py
import plotly.graph_objs as go

# ------ Import course toolbox ------
# from tmgsimple import TmgSimple

# ------ Import data ----- #
dir = os.getcwd()
doc = dir + '/battle-expert/datasheet/Concrete_Data.xls'
concreteList = pd.read_excel(doc)
concreteList.columns = [
    'Cement', 'Slag', 'Ash', 'Water', 'Superplasticiziser', 'Coarse Aggregate',
    'Fine Aggregate', 'Age', 'Strenght']

# Extract attribute names (1st row, column 1 to 9)
attributeNames = concreteList.head(0)
pd.set_option('precision', 1)
concreteList['Age'].describe()

# Summary statistics of the compressive strenght
concreteStrength = concreteList['Strenght']
concreteStrength.describe()

# Calculate quantiles
concreteStrengthQuantiles = concreteStrength.quantile([.2, .4, .6, .8])
concreteStrengthQuantiles

# Create Strenght catagory column based on conditions
conditions = [
    (concreteList['Strenght'] < concreteStrengthQuantiles.iloc[0]),
    (concreteList['Strenght'] > concreteStrengthQuantiles.iloc[0]) &
    (concreteList['Strenght'] < concreteStrengthQuantiles.iloc[1]),
    (concreteList['Strenght'] > concreteStrengthQuantiles.iloc[2]) &
    (concreteList['Strenght'] < concreteStrengthQuantiles.iloc[3]),
    (concreteList['Strenght'] > concreteStrengthQuantiles.iloc[3])
    ]
choices = ['Very Low', 'Low', 'High', 'Very High']
concreteList['Strenght Catagory'] = np.select(
    conditions, choices, default='Medium')

# Summary of the new catagory
concreteList['Strenght Catagory'].value_counts()

# ---- Attribute plots ---- #
# concreteList['Cement'].mean()

trace0 = go.Histogram(
    x=concreteList.iloc[:, 7],
    name='Age',
    showlegend=True
)
data = [trace0]
layout = go.Layout(showlegend=True, title='Age Histogram')
fig = go.Figure(data=data, layout=layout)
offline.plot(fig, filename='Histogram.html')

trace0 = go.Box(
    y=concreteList.iloc[:, 7],
    name='Age',
    showlegend=True
)
data = [trace0]
layout = go.Layout(showlegend=True, title='Age Boxplot')
fig = go.Figure(data=data, layout=layout)
offline.plot(fig, filename='Boxplot.html')

# Extract class names to python list,
# then encode with integers (dict)
classLabels = concreteList['Strenght Catagory']
classNames = sorted(set(classLabels))
classDict = dict(zip(classNames, range(5)))

# Compute values of N and M.
N = concreteList.shape[0]
M = concreteList.shape[1]
C = len(classNames)

# Extract vector y, convert to np matrix and transpose
y = np.mat([classDict[value] for value in classLabels]).T
X = np.array(concreteList.iloc[:, 0:-2])

# Data attributes to be plotted
i = 1
j = 2

# attribute names, and a title.
f = figure()
title('NanoNose data')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y.A.ravel() == c
    plot(X[class_mask, i], X[class_mask, j], '.')
legend(classNames)

# Substract the mean
Y = X - np.ones((N, 1)) * X.mean(0)

# PCA by computing SVD of Y
U, S, V = svd(Y, full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

# Plot variance explained
trace0 = go.Scatter(
    x=9,
    y=rho,
    mode='lines+markers',
    )
data = [trace0]
layout = go.Layout(
    title='Variance explained by principal components',
    yaxis=dict(title='Variance explained [%]'),
    xaxis=dict(title='Principal component'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig, filename='Boxplot.html')
