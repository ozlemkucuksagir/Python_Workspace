import numpy as np
import pandas as pd
import tensorflow as tf
#%%
print(tf.__version__)
#%%
data=pd.read_csv("ccpp.csv")
print(data.head())
#%%
print(data.describe())
#%%
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True,cmap="coolwarm")
#%%
print(data.isnull().sum())
#%%
data=data.dropna()
#%%
from sklearn.model_selection import train_test_split
x=data.iloc[:,:-1]
y=data.iloc[:,-1].values
xtrain,xtest,ytrain,ytest=train_test_split(x,y, test_size=0.2,random_state=22)
#%%
from tensorflow.keras.models import Sequential
ann=Sequential()
#%%
from tensorflow.keras.layers import Dense,Input
# giriş verileri (4 * 6) + bir bias = 24 + 6 =30
ann.add(Input(shape=(4,)))
# giriş verileri (6 * 8) + bir bias = 48 + 8 =56
ann.add(Dense(units=6,activation="relu"))
# giriş verileri (8 * 1) + bir bias = 8 + 1 =9
ann.add(Dense(units=8,activation="relu"))
ann.add(Dense(units=1,activation="linear"))
#%%
ann.compile(optimizer="adam",loss="mean_squared_error")
#%%
ann.summary()
#%%
history=ann.fit(xtrain,ytrain,epochs=100,batch_size=16,validation_data=(xtest,ytest))
#%%
ypred=ann.predict(xtest)
#%%
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(ytest,ypred)
r2=r2_score(ytest,ypred)
print("mse değeri:",mse)
print("r2 değeri",r2)
#%%
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss grafiği")
plt.xlabel("bağımsız değişken")
plt.ylabel("elektrik tüketimi")
plt.legend(["loss","val_loss"])
plt.show()
#%%
ann.save("model1.keras")
#%%
loaded_model=tf.keras.models.load_model("model1.keras")
#%%
manuel_data=np.array([[15.0,40.0,1015.0,85.0]])
prediction_output=loaded_model.predict(manuel_data)
#%%
print("tahmini değer:",prediction_output)





