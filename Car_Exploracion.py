# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:50:24 2020

@author: mario
"""

import os
import pandas as pd


directorio = 'C:\\Users\\mario\\OneDrive\\Bootcamp\\Proyectos\\DontGetKicked\\'
os.chdir(directorio)

data= pd.read_csv('Dataset\\train.csv')

train = data.copy()

from Scripts.Functions import *
train1= data_format(train)

train1.head()



# =============================================================================
# Exploring the labels
# =============================================================================

train.shape
train.columns
train.dtypes

sum(y)/len(y)


for col in train:
    print( (col), train[col].unique())

train1.isnull().sum()/train1.isnull().count()


# Univariate analysis
# =============================================================================

import seaborn as sns
sns.set(rc={'figure.figsize':(10,6)})


x= train1.Make
x= train1.Model_mod
x= train1.Trim
x= train1.SubModel
x= train1.Color
x= train1.Transmission
x= train1.WheelTypeID
x= train1.WheelType
x= train1.Size

x= train1.PurchDate
x= train1.VehYear
x= train1.VehOdo

x= train1.MMRAcquisitonRetailCleanPrice_avg_model


#   Numeric labels

sns.distplot(x)
sns.boxplot(x)
sns.boxenplot(x)
x.describe()


#   Categorical labels

sns.countplot(x)




# Bivariate analysis
# =============================================================================

import matplotlib.pyplot as plt

y= train.IsBadBuy

sns.lineplot(x,y)
sns.jointplot(x, y, kind="kde")


sns.relplot(x="VehYear",
            y="VehOdo", 
            hue="IsBadBuy",
            #s=50,
            kind="line",
            data=train1)


sns.relplot(x,y,sizes=(15, 200), data=train1)

corrmat = train1.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

corr = train1.corr()
corr.style.background_gradient(cmap='coolwarm')

