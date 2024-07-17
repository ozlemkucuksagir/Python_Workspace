import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Input
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
#%%
(xtrain,ytrain),(xtest,ytest)=fashion_mnist.load_data()
#%%
a=xtrain[0]
#%%
xtrain=xtrain/255
xtest=xtest/255
#%%
xtrain=np.expand_dims(xtrain, axis=-1)
xtest=np.expand_dims(xtest, axis=-1)
#%%
ytrain=to_categorical(ytrain,10)
ytest=to_categorical(ytest,10)
#%%
model=Sequential()
model.add(Input(shape=(28,28,1)))
model.add(Flatten())
#%%
model.add(Dense(128,activation="relu"))
model.add(Dense(64,activation="relu"))
model.add(Dense(10,activation="softmax"))
#%%
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
#%%
model.summary()
#%%
history=model.fit(xtrain,ytrain,epochs=100,batch_size=32,validation_data=(xtest,ytest))
#%%
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("loss grafiği")
plt.xlabel("loss")
plt.ylabel("epochs")
plt.legend(["loss","vallloss"])
plt.show()
#%%
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("acc grafiği")
plt.xlabel("acc")
plt.ylabel("epochs")
plt.legend(["acc","valacc"])
plt.show()
#%%
img_path="f1.png"
img=image.load_img(img_path,color_mode="grayscale",target_size=(28,28))
img_array=image.img_to_array(img)
img_array=img_array/255
#%%
img_array=np.expand_dims(img_array,axis=0)
#%%

