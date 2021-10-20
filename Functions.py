# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:08:05 2020

@author: mario
"""
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


# Data Processing
def data_format(dt):
    
    # Label preparation
    dt.Make= dt.Make.str.lower()
    dt.Model= dt.Model.str.lower()
    dt.Trim= dt.Trim.str.lower()
    dt.SubModel= dt.SubModel.str.lower()
    dt.Color= dt.Color.str.lower()
    dt.Transmission= dt.Transmission.str.lower()
    dt.WheelType= dt.WheelType.str.lower()
    dt.Size= dt.Size.str.lower()
    dt.TopThreeAmericanName= dt.TopThreeAmericanName.str.lower()
    dt.WheelTypeID = dt.WheelTypeID.astype(str)
    dt.WheelTypeID = dt.WheelTypeID.str.replace('.0','')
    dt.Model = dt.Model.str.replace('/','')
    dt.Model = dt.Model.str.replace('  ',' ')
    dt.Trim = dt.Trim.str.replace(' ','')
    dt.Trim = dt.Trim.str.replace('/','')
    dt.Trim = dt.Trim.str.replace('-','')
    dt.SubModel = dt.SubModel.str.replace('/','')
    dt['Make'] = dt.Make.replace('toyota scion','scion')
    dt.Nationality= dt.Nationality.str.lower()
    dt.Auction= dt.Auction.str.lower()
    dt.Nationality = dt.Nationality.astype(str)
    dt['Model_mod'] = dt['Model'].str.split().str[0]
    s= dt.Model_mod.value_counts()
    dt['Model_mod'] = np.where(dt['Model_mod'].isin(s.index[s < 100]), 'other', dt['Model_mod'])
    s= dt.Trim.value_counts()
    dt['Trim'] = np.where(dt['Trim'].isin(s.index[s < 500]), 'other', dt['Trim'])
    dt.Color = dt.Color.astype(str)
    
    # NaN treatment
    dt['Trim'] = dt['Trim'].fillna('nan')
    dt['SubModel'] = dt['SubModel'].fillna('nan')
    dt['Color'] = dt['Color'].fillna('nan')
    dt['Transmission'] = dt['Transmission'].fillna('nan')
    dt['WheelType'] = dt['WheelType'].fillna('nan')
    dt['Size'] = dt['Size'].fillna('nan')
    dt['Nationality'] = dt['Nationality'].fillna('nan')
    dt['TopThreeAmericanName'] = dt['TopThreeAmericanName'].fillna('nan')
    dt['Color'] = dt.Color.replace('nan','other')
    dt['Color'] = dt.Color.replace('not avail','other')
    dt['Nationality'] = dt.Nationality.replace('nan','american')
    dt['AUCGUART'] = dt['AUCGUART'].fillna('other')
    dt['VehOdo'].fillna((dt['VehOdo'].mean()), inplace=True)
    #dt['VehOdo'] = pd.qcut(dt['VehOdo'], q=3, labels=['low','medium','high'])
    
    # de momento quito los nans
    #dt.dropna(inplace=True)
    
    # Droping correlated and labels with 95% missings
    dt.drop('RefId', axis=1, inplace = True)
    dt.drop('Model',axis= 1, inplace= True)
    dt.drop('SubModel',axis= 1, inplace= True)
    dt.drop('PurchDate',axis=1, inplace= True)
    dt = dt.drop('VehicleAge', axis=1)
    #dt.drop('PRIMEUNIT', axis=1, inplace = True)

    # Treatment of price labels
    prices(dt)
    return dt



# ROC metric + Plot
def ROC(model, X, Y):
    probs = model.predict_proba(X)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(Y, preds)
    roc_auc = metrics.auc(fpr, tpr)
#    plt.title('Receiver Operating Characteristic')
#    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
#    plt.legend(loc = 'lower right')
#    plt.plot([0, 1], [0, 1],'r--')
#    plt.xlim([0, 1])
#    plt.ylim([0, 1])
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    plt.show()
    print ('Exact AUC:',roc_auc)

def plot_confusion_matrix(Y, y_hat):
    cf_matrix= (confusion_matrix(Y, y_hat))
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                         cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


# Metrics Multiple-in-one
def metric(model, X, Y):
    y_hat=model.predict(X)
    ROC(model, X, Y)
    print('F2-Score:', fbeta_score(Y, y_hat, average='binary', beta=2))
    print(classification_report(Y, y_hat))
    print(confusion_matrix(Y, y_hat))
    #plot_confusion_matrix(Y, y_hat)


def prices(dt):
    dt['DiffAuctionAv'] = (dt.MMRCurrentAuctionAveragePrice - dt.MMRAcquisitionAuctionAveragePrice)
    dt['DiffAuctionClean'] = (dt.MMRCurrentAuctionCleanPrice - dt.MMRAcquisitionAuctionCleanPrice)
    dt['DiffRetailAv'] = (dt.MMRCurrentRetailAveragePrice - dt.MMRAcquisitionRetailAveragePrice)
    dt['DiffRetailClean'] = (dt.MMRCurrentRetailCleanPrice - dt.MMRAcquisitonRetailCleanPrice)
    dt['Dif_MaxRetailSale_AcquisitionPrice'] = (dt.MMRCurrentRetailCleanPrice - dt.MMRAcquisitionAuctionAveragePrice)
    dt['Dif_RetailSale_AcquisitionPrice'] = (dt.MMRCurrentRetailAveragePrice - dt.MMRAcquisitionAuctionAveragePrice)
    dt['BPAuctionAv'] = ((dt.MMRCurrentAuctionAveragePrice*100)/dt.MMRAcquisitionAuctionAveragePrice)-100
    dt['BPAuctionClean'] = ((dt.MMRCurrentAuctionCleanPrice*100)/dt.MMRAcquisitionAuctionCleanPrice)-100
    dt['BPRetailAv'] = ((dt.MMRCurrentRetailAveragePrice*100)/dt.MMRAcquisitionRetailAveragePrice)-100
    dt['BPRetailClean'] = ((dt.MMRCurrentRetailCleanPrice*100)/dt.MMRAcquisitonRetailCleanPrice)-100
    price = dt[['MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice','Dif_MaxRetailSale_AcquisitionPrice','Dif_RetailSale_AcquisitionPrice','DiffAuctionAv','DiffAuctionClean','DiffRetailAv','DiffRetailClean', 'BPAuctionAv','BPAuctionClean','BPRetailAv','BPRetailClean']]
    for col in price:
        dt[col+"_avg_model"]=dt[col].fillna(dt.groupby(['Model_mod'])[col].transform('mean'))
    #price_avg_model= dt[['MMRAcquisitionAuctionAveragePrice_avg_model', 'MMRAcquisitionAuctionCleanPrice_avg_model', 'MMRAcquisitionRetailAveragePrice_avg_model', 'MMRAcquisitonRetailCleanPrice_avg_model', 'MMRCurrentAuctionAveragePrice_avg_model', 'MMRCurrentAuctionCleanPrice_avg_model', 'MMRCurrentRetailAveragePrice_avg_model', 'MMRCurrentRetailCleanPrice_avg_model', 'Dif_MaxRetailSale_AcquisitionPrice_avg_model', 'Dif_RetailSale_AcquisitionPrice_avg_model', 'DiffAuctionAv_avg_model', 'DiffAuctionClean_avg_model', 'DiffRetailAv_avg_model', 'DiffRetailClean_avg_model']]
    #for col in price_avg_model:
        #dt[col] = pd.qcut(dt[col], q=3, labels=['low','medium','high'])