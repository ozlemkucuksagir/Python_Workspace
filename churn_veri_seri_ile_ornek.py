import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
data=pd.read_csv("Churn_Modelling.csv")
#%%
x=data.iloc[:,3:-1].values
y=data.iloc[:,-1].values
#%%
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
#%%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[("nation",OneHotEncoder(),[1])],remainder="passthrough")
#%%
x=np.array(ct.fit_transform(x))
#%%
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y, test_size=0.2,random_state=22)
#%%
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
#%%
from tensorflow.keras.models import Sequential
model=Sequential()
#%%
from tensorflow.keras.layers import Dense,Input
model.add(Input(shape=(12,)))

model.add(Dense(units=24,activation="relu"))

model.add(Dense(units=6,activation="relu"))
model.add(Dense(units=1,activation="sigmoid"))
#%%
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#%%
model.summary()
#%%
history=model.fit(xtrain,ytrain,batch_size=16,epochs=100,validation_data=(xtest,ytest))
#%%
import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model")
plt.xlabel("harcama puanı")
plt.ylabel("sistemden ayrıma durumu")
plt.legend(["Train","Val-Train"])
plt.show()
#%%
import matplotlib.pyplot as plt
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model")
plt.xlabel("harcama puanı")
plt.ylabel("sistemden ayrıma durumu")
plt.legend(["loss","Val-loss"])
plt.show()

#%%
"""from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(xtrain,ytrain)
ypred=knn.predict(xtest)
knn.score(xtest,ytest)"""
#%%
tahmin=model.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))
#%%
print(tahmin>0.5)
#%%
ypred=model.predict(xtest)
#%%
ypred = (ypred > 0.5)
#%%
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(ytest,ypred)
print(cm)
print(accuracy_score(ytest,ypred))





