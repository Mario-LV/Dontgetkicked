# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:50:54 2020

@author: mario
"""

import os
import pandas as pd

def combine(path):
    os.chdir(path)    
    math= pd.read_csv('student-mat.csv')
    port= pd.read_csv('student-por.csv')
    train_combined= pd.concat([math,port])
    train_combined.to_csv('train_combined.csv')

combine('C:\\Users\\mario\\OneDrive\\Bootcamp\\Proyectos\\\Alcohol_students\\')


directorio = 'C:\\Users\\mario\\OneDrive\\Bootcamp\\Proyectos\\'
os.chdir(directorio)

math= pd.read_csv('HR.csv')
port= pd.read_csv('student-por.csv')

math.shape

math['Asignatura']= 'math'
port['Asignatura']= 'port'

first_comb= pd.concat([math,port],sort=True)

individual = first_comb[["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]]
duplicatedIndividual = individual[individual.duplicated(keep='first')]
print(duplicatedIndividual)

comb_ind= individual.drop_duplicates()

comb_ind.head()




combined= first_comb.drop(duplicatedIndividual)

dataset.to_csv('train_combined.csv')

