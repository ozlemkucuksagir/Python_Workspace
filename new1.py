import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/ozlem/Desktop/Python_Workspace/Churn_Modelling.csv")

x = data.iloc[:, 3:-1].values
y = data.iloc[:, -1].values
#%%
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# Sadece kategorik değişkenler için etiket kodlaması uygulayın
x[:, 1] = le.fit_transform(x[:, 1])
x[:, 2] = le.fit_transform(x[:, 2])
#%%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformer=[("nation",OneHotEncoder()[1])],remainder="passthrough")

#%%
x=np.array(ct.fit_transform(x))
