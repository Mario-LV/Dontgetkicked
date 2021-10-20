# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:51:26 2020

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
# Data Preprocessing
# =============================================================================

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 1000)

# =============================================================================
# MMRAcquisitionAuctionAveragePrice        Precio promedio momento compra                             PromCompra
# MMRAcquisitionAuctionCleanPrice          Precio superiores momento compra                           SupCompra
# MMRAcquisitionRetailAveragePrice         Precio promedio de segunda mano momento compra             Prom2Compra
# MMRAcquisitonRetailCleanPrice            Precio superior de segunda mano momento compra             Sup2Compra
# MMRCurrentAuctionAveragePrice            Precio promedio Actual                                     PromActual
# MMRCurrentAuctionCleanPrice              Precio superior actual                                     SupActual
# MMRCurrentRetailAveragePrice             Precio promedio segunda mano actual                        Prom2Actual
# MMRCurrentRetailCleanPrice               Precio superior segunda mano actual                        Sup2Actual
# =============================================================================

price = train1[['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice','Dif_MaxRetailSale_AcquisitionPrice','Dif_RetailSale_AcquisitionPrice','DiffAuctionAv','DiffAuctionClean','DiffRetailAv','DiffRetailClean']]

price.describe()

price.astype(int)
# Columna menor precio medio: MMRAcquisitionAuctionAveragePrice apenas varia con MMRCurrentAuctionAveragePrice
# Columna mayor precio medio: MMRCurrentRetailCleanPrice aoenas varia con MMRAcquisitonRetailCleanPrice

# El menor precio de compra se encuentra en la PromCompra y apenas varia en PromActual.
# El mayor precio de compra se encuentra en la Sup2Actual y apenas varia con el de Sup2Compra


# Por lo tanto Sup2Actual - PromCompra. Nos dara la diferencia maxima que puede haber entre el precio de compra y el precio actual en el mercado de segunda mano
# Tiene sentido comparar precio de coches promedio con superiores? Mejor compararlo con el valor promedio actual en mercado segunda mano. Prom2Actual - PromCompra


train1['Dif_MaxRetailSale_AcquisitionPrice'] = (train1.MMRCurrentRetailCleanPrice - train1.MMRAcquisitionAuctionAveragePrice)
train1['Dif_RetailSale_AcquisitionPrice'] = (train1.MMRCurrentRetailAveragePrice - train1.MMRAcquisitionAuctionAveragePrice)


# Comparo cada precio de venta con el de equivalente de compra.

train1['DiffAuctionAv'] = (train1.MMRCurrentAuctionAveragePrice - train1.MMRAcquisitionAuctionAveragePrice)
train1['DiffAuctionClean'] = (train1.MMRCurrentAuctionCleanPrice - train1.MMRAcquisitionAuctionCleanPrice)
train1['DiffRetailAv'] = (train1.MMRCurrentRetailAveragePrice - train1.MMRAcquisitionRetailAveragePrice)
train1['DiffRetailClean'] = (train1.MMRCurrentRetailCleanPrice - train1.MMRAcquisitonRetailCleanPrice)

# Obtengo el porcentaje de beneficio

train1['BPAuctionAv'] = ((train1.MMRCurrentAuctionAveragePrice*100)/train1.MMRAcquisitionAuctionAveragePrice)-100
train1['BPAuctionClean'] = ((train1.MMRCurrentAuctionCleanPrice*100)/train1.MMRAcquisitionAuctionCleanPrice)-100
train1['BPRetailAv'] = ((train1.MMRCurrentRetailAveragePrice*100)/train1.MMRAcquisitionRetailAveragePrice)-100
train1['BPRetailClean'] = ((train1.MMRCurrentRetailCleanPrice*100)/train1.MMRAcquisitonRetailCleanPrice)-100



# NaNs

train1.isnull().sum()/train1.isnull().count()

# Localizar el 'Model' de las filas NaN
# Agrupar todas las filas con ese mismo model
# Hacer la media de cada columna de precio del conjunto de filas
# Cambiar los NaN con esa media


train1['MMRAcquisitionAuctionAveragePrice'] = train1.groupby(['Model']).MMRAcquisitionAuctionAveragePrice.apply(lambda x: x.fillna(x.mean()))

# te has asegurado que todos los modelos tienen almenos un precio puesto? 



train1.groupby(['Model'])['MMRAcquisitionAuctionAveragePrice'].transform('mean')
train1.groupby(['Model'])['MMRAcquisitionAuctionAveragePrice'].transform('var')

# Creo nueva variable con el precio medio de cada modelo de coche para rellenar los nans
for col in price:
    train1[col+"_avg_model"]=train1[col].fillna(train1.groupby(['Model_mod'])[col].transform('mean'))


#price_avg_model= train1[['MMRAcquisitionAuctionAveragePrice_avg_model','MMRAcquisitionAuctionCleanPrice_avg_model','MMRAcquisitionRetailAveragePrice_avg_model','MMRAcquisitonRetailCleanPrice_avg_model', 'MMRCurrentAuctionAveragePrice_avg_model', 'MMRCurrentAuctionCleanPrice_avg_model', 'MMRCurrentRetailAveragePrice_avg_model', 'MMRCurrentRetailCleanPrice_avg_model', 'Dif_MaxRetailSale_AcquisitionPrice_avg_model', 'Dif_RetailSale_AcquisitionPrice_avg_model', 'DiffAuctionAv_avg_model', 'DiffAuctionClean_avg_model', 'DiffRetailAv_avg_model', 'DiffRetailClean_avg_model']]
#
#
#for col in price_avg_model:
#    train1[col] = pd.qcut(train1[col], q=3, labels=['low','medium','high'])



# =============================================================================
# Make					Vehicle Manufacturer
# Model					Vehicle Model
# Trim					Vehicle Trim Level
# SubModel				Vehicle Submodel
# Color					Vehicle Color
# Transmission			Vehicles transmission type (Automatic, Manual)
# WheelTypeID			The type id of the vehicle wheel
# WheelType				The vehicle wheel type description (Alloy, Covers)
# Size					The size category of the vehicle (Compact, SUV, etc.)
# TopThreeAmericanName	Identifies if the manufacturer is one of the top three American manufacturers
# =============================================================================


# Tratamiento General
# =============================================================================
train1.Size.unique()

train1.Make= train1.Make.str.lower()
train1.Model= train1.Model.str.lower()
train1.Trim= train1.Trim.str.lower()
train1.SubModel= train1.SubModel.str.lower()
train1.Color= train1.Color.str.lower()
train1.Transmission= train1.Transmission.str.lower()
train1.WheelType= train1.WheelType.str.lower()
train1.Size= train1.Size.str.lower()
train1.TopThreeAmericanName= train1.TopThreeAmericanName.str.lower()


train1.Model = train1.Model.str.replace('/','')
train1.Model = train1.Model.str.replace('  ',' ')

train1.Trim = train1.Trim.str.replace(' ','')
train1.Trim = train1.Trim.str.replace('/','')
train1.Trim = train1.Trim.str.replace('-','')

train1.SubModel = train1.SubModel.str.replace('/','')
train1.SubModel = train1.SubModel.str.replace('-','')

train1.WheelTypeID = train1.WheelTypeID.astype(str)
train1.WheelTypeID = train1.WheelTypeID.str.replace('.0','')
# train1.WheelTypeID = train1.WheelTypeID.str.replace('\d+', '')

train1['Trim'] = train1['Trim'].fillna('nan')
train1['SubModel'] = train1['SubModel'].fillna('nan')
train1['Color'] = train1['Color'].fillna('nan')
train1['Transmission'] = train1['Transmission'].fillna('nan')
train1['WheelType'] = train1['WheelType'].fillna('nan')
train1['Trim'] = train1['Trim'].fillna('nan')
train1['Size'] = train1['Size'].fillna('nan')
train1['Nationality'] = train1['Nationality'].fillna('nan')
train1['TopThreeAmericanName'] = train1['TopThreeAmericanName'].fillna('nan')

train1.isnull().sum()/train1.isnull().count()

null_columns=train1.columns[train1.isnull().any()]
train1[null_columns].isnull().sum()
print(train1[train1.isnull().any(axis=1)][null_columns].head())


# Tratamiento Make
# =============================================================================
train1.Make.unique()
train1.Make.value_counts()

train1['Make'] = train1.Make.replace('toyota scion','scion')

# Elimino todas las filas en las que 'Make' tenga menos de 100 registros en comun.

# =============================================================================
# train1.drop(train1.loc[train1['Make'] == 'hummer'].index, inplace=True)
# train1.drop(train1.loc[train1['Make'] == 'plymouth'].index, inplace=True)
# train1.drop(train1.loc[train1['Make'] == 'mini'].index, inplace=True)
# train1.drop(train1.loc[train1['Make'] == 'subaru'].index, inplace=True)
# train1.drop(train1.loc[train1['Make'] == 'lexus'].index, inplace=True)
# train1.drop(train1.loc[train1['Make'] == 'cadillac'].index, inplace=True)
# train1.drop(train1.loc[train1['Make'] == 'acura'].index, inplace=True)
# train1.drop(train1.loc[train1['Make'] == 'volvo'].index, inplace=True)
# train1.drop(train1.loc[train1['Make'] == 'infiniti'].index, inplace=True)
# train1.drop(train1.loc[train1['Make'] == 'lincoln'].index, inplace=True)
# =============================================================================

# Agrupar 'Make' con menos de 100 registros a la que más se asemejen los precios.
# train1[train1['Make'].str.contains("scion")].groupby(["IsBadBuy","Model"]).size()


# Tratamiento Model
# =============================================================================
train1.Model_mod.unique()
import numpy as np

train1['Model_mod'] = train1['Model'].str.split().str[0]

s= train1.Model_mod.value_counts()
train1['Model_mod'] = np.where(train1['Model_mod'].isin(s.index[s < 100]), 'other', train1['Model_mod'])

# train1[train1['Model_mod'].str.contains("other")].groupby(['Model_mod',"IsBadBuy"]).size()


# Tratamiento Submodel
# =============================================================================

train1.SubModel.unique()
train1.SubModel.value_counts()

train1.drop('SubModel',axis= 1, inplace= True)


# Tratamiento Trim
# =============================================================================
null_columns=train1.columns[train1.isnull().any()]
train1[null_columns].isnull().sum()
print(train1[train1.isnull().any(axis=1)][null_columns].head())


print(train1[train1["Trim"].isnull()][null_columns])

s= train1.Trim.value_counts()
train1['Trim'] = np.where(train1['Trim'].isin(s.index[s < 500]), 'other', train1['Trim'])

train1['Trim'].value_counts()
np.sum(train1.Trim.value_counts()>500)

# Tratamiento Color
# =============================================================================
# Group unknown colours in 'other'
train1.Color.unique()

train1.Color = train1.Color.astype(str)
train1['Color'] = train1.Color.replace('nan','other')
train1['Color'] = train1.Color.replace('not avail','other')

train1.Color.value_counts()





# =============================================================================
# PurchDate				The Date the vehicle was Purchased at Auction
# VehYear				The manufacturer's year of the vehicle
# VehicleAge			The Years elapsed since the manufacturer's year
# =============================================================================

train1.PurchDate.unique()

train1.drop('PurchDate',axis=1, inplace= True)

train1.WheelTypeID = train1.WheelTypeID.astype(str)
train1.WheelTypeID = train1.WheelTypeID.str.replace('.0','')


train1 = train1.drop('VehicleAge', axis=1) #por el momento



# =============================================================================
# VehOdo				The vehicles odometer reading
# Nationality			The Manufacturer's country
# Auction					Auction provider at which the  vehicle was purchased
# =============================================================================

train1.Nationality.unique()
train1.Nationality= train1.Nationality.str.lower()
train1.Auction= train1.Auction.str.lower()
train1.Nationality = train1.Nationality.astype(str)

train1['Nationality'] = train1.Nationality.replace('nan','american')



pd.crosstab(train1.Nationality, train1.IsBadBuy, normalize ='index')

# Marco los cortes segun 25%-50%-75% de .describe()
train1['VehOdo'] = pd.cut(train1['VehOdo'], bins=[0, 61837, 73361, 82436], labels=['low','medium','high'])
train1['VehOdo'].fillna((train1['VehOdo'].mean()), inplace=True)

pd.crosstab(train1.PRIMEUNIT, train1.AUCGUART)



# PRIMEUNIT eliminar. Escribir motivos:
# =============================================================================
# Pendientes para perfeccionar
# =============================================================================

# Agrupar los modelos con menos de 50 registros junto con la misma variable de ese modelo que este menos detallada. Pej: pasar 'sentra 2.5L' a 'sentra', pero la 'sentra 1.8L', que tiene mas de 50 registros, dejarla como esta.

# =============================================================================
# # train1.Model_mod= train1.Model.value_counts()# <=50 'other'. En 3 pasos. Definir, crear lista, replace con for. Ver como añadirlo a lo mas parecido al registro
#    model_mod = list(train1.Model.value_counts() <= 50)
#    for value in len(train1.Model_mod):
#       train1.Model.str.replace('',value)
# =============================================================================
# =============================================================================
# #Creo titulo a partir de nombre y elimino nombre
# 
# model_mod = (train1.Model.value_counts() <= 50).index
# train1.model_mod = model_mod
# #train1['Model']=0
# for i in train1:
#     train1['Model']=train1['model_mod'].str.extract('([A-Za-z]+)\.', expand=False)  # Use REGEX to define a search pattern
# # train1.drop('Name', axis = 1, inplace = True)
# 
# 
# pd.crosstab(index=train1["Model"], columns="count")
# 
# # Si tengo un Miss, seguramente es alguien joven, pero le he imputado la misma edad que al resto.
# import seaborn as sns
# 
# #miramos estabilidad de la edad para cada categoria de Titulo. Vemos que etrain1cepto en Rev, Dr y Major, en general es muy estable
# #la barra negra pequeña nos indica que hay poca variabilidad de la edad en la categoría
# sns.barplot(train1=train1[~train1.index.isin(model_mod)]['Model'], y=train1[~train1.index.isin(model_mod)]['model_mod']);
# #es una buena forma de imputar los missings
# 
# # calculo edad media asociada a cada titulo 
# means = train1[~train1.index.isin(model_mod)][["model_mod", "Model"]].groupby("Model").mean()
# # means = train1.groupby('Title')['Age'].mean() #Asi no estaría bien calculado, utilizaria los ya imputados para sacar la media!
# map_means = means["model_mod"].to_dict()
# 
# #imputo
# train1.loc[model_mod,'model_mod'] = train1['Model'].loc[model_mod].map(map_means)
# =============================================================================
