from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import HuberRegressor
import pandas as pd
import numpy as np
# I considered two targets (price and vehicle trim)simultaneously.
A1 = pd.read_csv('Training_DataSet.csv')
A2 = pd.read_csv('Test_DataSet.csv')
A1 = A1[:6300]
lbl = A1.loc[:,'Dealer_Listing_Price'].tolist()
lbl2 = A1.loc[:,'Vehicle_Trim'].tolist()

A1.replace(np.nan, -1,inplace = True)
A2.replace(np.nan, -1,inplace = True)
cat = ['ListingID','SellerCity', 'SellerIsPriv','SellerListSrc',
       'SellerName','SellerRating','SellerRevCnt','SellerState','SellerZip',
       'VehBodystyle','VehCertified','VehColorExt','VehColorInt','VehDriveTrain',
       'VehEngine','VehFeats','VehFuel','VehHistory','VehListdays','VehMake',
       'VehMileage', 'VehModel','VehPriceLabel','VehSellerNotes','VehType',
       'VehTransmission', 'VehYear']
At1= pd.get_dummies(A1,columns = cat)
At2= pd.get_dummies(A2,columns = cat)

train_x = At1
train_y = lbl
test_x = At2
#train_X = np.array(At1.values.tolist())
#train_y = np.array(lbl.values.tolist())
#test_x = np.array(A2.values.tolist())
scaler = StandardScaler()
scaler.fit(train_x)
train_img = scaler.transform(train_x)
test_img = scaler.transform(test_x)

pca = PCA(.90)
pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
pipeline = Pipeline(steps=[('normalize', StandardScaler()), ('model', HuberRegressor())])

model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())

cv = KFold(n_splits=10, shuffle=True, random_state=1)
scores = cross_val_score(model, train_x, train_y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

expected = lbl
predicted = model.predict(test_x)
print("Accuracy: ", r2_score(expected, predicted))
