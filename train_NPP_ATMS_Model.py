
import cPickle as pickle
from numpy import *
dbL1=pickle.load(open('npp_atms_w0_Ocean.pklz','rb'))
dbL1=array(dbL1)

import keras
from keras.layers import Activation

n_input=10

X=dbL1[:,0:n_input]
Y=dbL1[:,-1]
Y2=[]
ic=0.
ict=0
for y in Y:
    ict+=1
    if y>0:
        Y2.append([1,0])
        ic+=1.
    else:
        Y2.append([0,1])
print ic/ict

Y2=array(Y2)
a=nonzero(Y>=0.01)
X=X[a[0],:]
Y=Y[a]
a=nonzero(abs(X[:,-2]-48)<40)
X=X[a[0],:]
Y=Y[a]


r=random.random(X.shape[0])
a=nonzero(r<0.5)
b=nonzero(r>0.5)


model = keras.Sequential()
model.add(keras.layers.Dense(64,input_shape=(n_input,)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(32,activation='sigmoid'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(32,activation='sigmoid'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam")

model.fit(X[a[0],:],(Y[a]),batch_size=1000,\
          verbose=2,epochs=50)


y=(model.predict(X[b[0],:]))
print corrcoef(y[:,0],Y[b])
c=nonzero(Y[b]>0.001)
print Y[b][c].mean(),y[:,0][c].mean()

model.save("keras_NPP_ATMS_pred_Ocean.h5")
