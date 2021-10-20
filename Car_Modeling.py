# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:53:14 2020

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
# Benchmark
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score, make_scorer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



Y= train1.IsBadBuy

X= train1[['Make', 'Model_mod','Auction','Nationality','Trim','VNST','VehBCost','IsOnlineSale','WarrantyCost','WheelType','WheelTypeID', 'VehOdo', 'VehYear', 'Color', 'Transmission', 'Size', 'AUCGUART', 'Dif_MaxRetailSale_AcquisitionPrice_avg_model', 'Dif_RetailSale_AcquisitionPrice_avg_model', 'DiffAuctionAv_avg_model', 'DiffAuctionClean_avg_model', 'DiffRetailAv_avg_model', 'DiffRetailClean_avg_model']]
X = pd.get_dummies(X, drop_first=True)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=7,train_size=0.7)


# Logistic Regression
# =============================================================================

clf = LogisticRegression(random_state=0, solver='lbfgs')
clf=clf.fit(X_train, Y_train)


# METRICS: Accuracy, Precision, recall, F-measure and support

metric(clf, X_test, Y_test)
metric(clf, X_train, Y_train)

import statsmodels.api as sm
X=sm.add_constant(X_train)
logit_model=sm.Logit(Y_train, X_train)
result=logit_model.fit()
print(result.summary())

X.shape

# Singlar matrix = Multicolinealidad perfecta. Revisar que no haya dos variables correlacionadas




# =============================================================================
# Modeling after data preprocessing
# =============================================================================

from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

Y= train1['IsBadBuy']


#train1.isnull().sum()/train1.isnull().count()
X = train1.drop('IsBadBuy', axis=1)

#X= train1[['Make', 'Model_mod','Auction','Nationality','VNST','VehBCost','IsOnlineSale','WarrantyCost', 'Trim','WheelType','WheelTypeID', 'VehOdo', 'VehYear', 'Color', 'Transmission', 'Size', 'AUCGUART', 'MMRAcquisitionAuctionAveragePrice_avg_model', 'MMRAcquisitionAuctionCleanPrice_avg_model', 'MMRAcquisitionRetailAveragePrice_avg_model', 'MMRAcquisitonRetailCleanPrice_avg_model', 'MMRCurrentAuctionAveragePrice_avg_model', 'MMRCurrentAuctionCleanPrice_avg_model', 'MMRCurrentRetailAveragePrice_avg_model', 'MMRCurrentRetailCleanPrice_avg_model', 'Dif_MaxRetailSale_AcquisitionPrice_avg_model', 'Dif_RetailSale_AcquisitionPrice_avg_model', 'DiffAuctionAv_avg_model', 'DiffAuctionClean_avg_model', 'DiffRetailAv_avg_model', 'DiffRetailClean_avg_model','BPAuctionAv_avg_model', 'BPAuctionClean_avg_model','BPRetailAv_avg_model','BPRetailClean_avg_model']]
X = pd.get_dummies(X, drop_first=True)

#X.shape

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=7,train_size=0.7)

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance

xgbmodel = XGBClassifier()
xgbmodel.get_params()

# parametros para variables seleccionadas
# =============================================================================
params={'base_score': [0.5], # prediccion inicial
     'booster': ['gbtree'], # (gbtree, gblinear, dart)
     'colsample_bylevel': [0.8], # ratio de columnas en cada nivel. 0.8
     'colsample_bytree': [0.8], # ratio de columnas por tree
     'gamma': [1],    # minimo "loss" reduccion para crear un nuevo split. Larger-> conservative. 1
     'learning_rate': [0.2], # (eta) aportacion de cada arbol al modelo. 0.3 --> 0.4
     'max_depth': [3], # maxima profundidad en cada arbol. 1 --> 3
     'min_child_weight': [1], # minimo numero samples por hoja
     'missing': [None], # si queremos reemplazar los missings por un numero
     'n_estimators': [90], # numero de arboles. 200 --> 100
     'n_jobs': [-1], # trabajos en paralelo
     'objective': ['binary:logistic'],#  Output. Tipo de función que estamos estimando
     'random_state': [0], # seed para generar los folds
     'reg_alpha': [0.5], # L1 regularitacion. 0.01 --> 1?
     'reg_lambda': [0.15], # L2 regularitacion. 0.01 --> 0.2?
     'scale_pos_weight': [9], # 1 -->9
     'subsample': [0.7]} # ratio de muestras por cada arbol
# =============================================================================

params={'base_score': [0.5], # prediccion inicial
     'booster': ['gbtree'], # (gbtree, gblinear, dart)
     'colsample_bylevel': [0.8], # ratio de columnas en cada nivel. 0.8
     'colsample_bytree': [0.8], # ratio de columnas por tree
     'gamma': [1],    # minimo "loss" reduccion para crear un nuevo split. Larger-> conservative. 1
     'learning_rate': [0.2], # (eta) aportacion de cada arbol al modelo. 0.3 --> 0.4
     'max_depth': [3], # maxima profundidad en cada arbol. 1 --> 3
     'min_child_weight': [1], # minimo numero samples por hoja
     'missing': [None], # si queremos reemplazar los missings por un numero
     'n_estimators': [90], # numero de arboles. 200 --> 100
     'n_jobs': [-1], # trabajos en paralelo
     'objective': ['binary:logistic'],#  Output. Tipo de función que estamos estimando
     'random_state': [0], # seed para generar los folds
     'reg_alpha': [0.5], # L1 regularitacion. 0.01 --> 1?
     'reg_lambda': [0.15], # L2 regularitacion. 0.01 --> 0.2?
     'scale_pos_weight': [10], # 1 -->9
     'subsample': [0.7]} # ratio de muestras por cada arbol

f2 = make_scorer(fbeta_score, beta=2)

grid_solver = GridSearchCV(estimator = xgbmodel, # model del train
                   param_grid = params,
                   scoring = f2,
                   cv = 5,
                   n_jobs=-1,
                   verbose = 2)

model_result_xgboost = grid_solver.fit(X_train,Y_train)

metric(model_result_xgboost, X_train, Y_train)
metric(model_result_xgboost, X_test, Y_test)

model_result_xgboost.best_params_

plot_importance(model_result_xgboost)

# Busco las variables más importantes del modelo
best_model = model_result_xgboost.best_estimator_
final_model = best_model.fit(X_train,Y_train)
len(X_train.columns)
len(final_model.feature_importances_)
importances=pd.DataFrame([X_train.columns,final_model.feature_importances_], index=["feature","importance"]).T
importances


importances= importances.sort_values(by = "importance", ascending = False).iloc[0:20,]

# model_result_xgboost.score(X_test,Y_test)

# =============================================================================
# Durante el finde seguir probando scale_pos y otros parametros
# =============================================================================

# Train: reg_a 1 y reg_l 0.2: F2-Score: 0.4181154075791101
# Test: reg_a 1 y reg_l 0.2: F2-Score: 0.33336100963135173

# Train: scale_pos 10: F2-Score: 0.59719546615295
# Test: scale_pos 10: F2-Score: 0.4906665260493567



# Train: scale_pos 1: F2-Score: 0.34960951755346326 [[44511   274]
#                                                   [ 4387  1916]]
# Test: scale_pos 1: F2-Score: 0.2823590363526466 [[18987   235]
#                                                 [ 2019   654]]

# Train: scale_pos 4: F2-Score: 0.5489927878637155 [[41361  3424]
#                                                  [ 2771  3532]]
# Test: scale_pos 4: F2-Score: 0.42600235849056606 [[17502  1720]
#                                                  [ 1517  1156]]

# Train: scale_pos 5: F2-Score: 0.587381318777039 [[39602  5183]
#                                                 [ 2257  4046]]
# Test: scale_pos 5: F2-Score: 0.4573547589616811 [[16684  2538]
#                                                 [ 1341  1332]]

# Train: scale_pos 10: F2-Score: 0.5969181222463266 [[30635 14150]
#                                                   [  974  5329]]
# Test: scale_pos 10: F2-Score: 0.4983897365503405 [[12856  6366]
#                                                  [  822  1851]]



# =============================================================================
# Train vs test. No se puede pk el test lo usan para evaluar y viene sin IsBadBuy.
# =============================================================================

# test
# =============================================================================
data1= pd.read_csv('Dataset\\test.csv')

test = data1.copy()

from Scripts.Functions import *
test1= data_format(test)

Yt= test1['IsBadBuy']

Xt= test1[['Make', 'Model_mod','Auction','Nationality','VNST','VehBCost','IsOnlineSale','WarrantyCost', 'Trim','WheelType','WheelTypeID', 'VehOdo', 'VehYear', 'Color', 'Transmission', 'Size', 'AUCGUART', 'MMRAcquisitionAuctionAveragePrice_avg_model', 'MMRAcquisitionAuctionCleanPrice_avg_model', 'MMRAcquisitionRetailAveragePrice_avg_model', 'MMRAcquisitonRetailCleanPrice_avg_model', 'MMRCurrentAuctionAveragePrice_avg_model', 'MMRCurrentAuctionCleanPrice_avg_model', 'MMRCurrentRetailAveragePrice_avg_model', 'MMRCurrentRetailCleanPrice_avg_model', 'Dif_MaxRetailSale_AcquisitionPrice_avg_model', 'Dif_RetailSale_AcquisitionPrice_avg_model', 'DiffAuctionAv_avg_model', 'DiffAuctionClean_avg_model', 'DiffRetailAv_avg_model', 'DiffRetailClean_avg_model','BPAuctionAv_avg_model', 'BPAuctionClean_avg_model','BPRetailAv_avg_model','BPRetailClean_avg_model']]
Xt = pd.get_dummies(X, drop_first=True)

X_test,Y_test = train_test_split(Xt,Yt,random_state=7)
