# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:08:33 2024

@author: ozlem
"""

import numpy as np
import pandas as pd
import tensorflow as tf

print(tf.__version__)

data=pd.read_csv("C:/Users/ozlem/Desktop/Python_Workspace/ccpp.csv")
print(data.head())
#½½
x=data.iloc[:,:-1]
y=data.iloc[:,-1].values
#½½

from sklearn.model_selection import train_test_split
xtrain,xtemp,ytrain,ytemp=train_test_split(x,y,test_size=0.4, random_state=35)
#½½
xtest,xval,ytest,yval=train_test_split(xtemp,ytemp,test_size=0.5, random_state=23)