import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
A1 = pd.read_csv('Training_DataSet.csv')
A2 = pd.read_csv('Test_DataSet.csv')
A1 = A1[:6300]
lbl = A1.loc[:,'Dealer_Listing_Price'].tolist()
cat1 = ['Vehicle_Trim']
lbl2 = A1.loc[:,'Vehicle_Trim']
lbt2 = pd.get_dummies (A1, columns= cat1, dtype = float)

A1.replace(np.nan, -1,inplace = True)
A2.replace(np.nan, -1,inplace = True)
cat = ['ListingID','SellerCity', 'SellerIsPriv','SellerListSrc',
       'SellerName','SellerRating','SellerRevCnt','SellerState','SellerZip',
       'VehBodystyle','VehCertified','VehColorExt','VehColorInt','VehDriveTrain',
       'VehEngine','VehFeats','VehFuel','VehHistory','VehListdays','VehMake',
       'VehMileage', 'VehModel','VehPriceLabel','VehSellerNotes','VehType',
       'VehTransmission', 'VehYear']
At1= pd.get_dummies(A1,columns = cat, dtype = float)
At2= pd.get_dummies(A2,columns = cat, dtype = float)
X_train = np.asarray(At1)
y_train = np.asarray(lbl)
X_test = np.asarray(At2)
y_test =np.asarray(lbl2)
y_train_1 = np.asarray(lbl)
y_train_2 = np.asarray(lbl2)

def get_model(input_dim):

    input_layer = keras.Input(shape=(input_dim,), name="input_layer")

    dense_1 = keras.layers.Dense(input_dim, name = 'dense_1')(input_layer)
    dense_2 = keras.layers.Dense(input_dim, name = 'dense_2')(dense_1)

    regression_output = keras.layers.Dense(1, activation = 'linear', name = 'regression_output')(dense_2)
    classification_output = keras.layers.Dense(1, activation = 'sigmoid', name = 'classification_output')(dense_2)
    model = keras.Model(inputs=input_layer,outputs=[regression_output, classification_output])
    
    return(model)
for loss_weight_param in ([1,100],[1,50],[1,10],[1,1],[10,1],[50,1],[100,1]):
    model = get_model(20)
    model.compile(
    optimizer="adam",
    loss=[
        keras.losses.MeanSquaredError(),
        keras.losses.BinaryCrossentropy(),
    ],loss_weights = loss_weight_param)
    
    model.fit(X_train,
    {"regression_output": y_train_1, "classification_output": y_train_2},
    epochs=10,
    batch_size=10,
          verbose=0)
    predictions = model.predict(model, X_test) 

