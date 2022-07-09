import pandas as pd
import numpy as np
A1 = pd.read_csv('Training_DataSet.csv')
A2 = pd.read_csv('Test_DataSet.csv')
A1 = A1[:6300]
cat = ['ListingID','SellerCity', 'SellerIsPriv','SellerListSrc',
       'SellerName','SellerRating','SellerRevCnt','SellerState','SellerZip',
       'VehBodystyle','VehCertified','VehColorExt','VehColorInt','VehDriveTrain',
       'VehEngine','VehFeats','VehFuel','VehHistory','VehListdays','VehMake',
       'VehMileage', 'VehModel','VehPriceLabel','VehSellerNotes','VehType',
       'VehTransmission', 'VehYear']

At1= pd.get_dummies(A1,columns = cat,dtype = float)
At2= pd.get_dummies(A2,columns = cat, dtype = float)
df_train = pd.read_csv('Training_DataSet.csv').drop('ListingID', axis=1)
df_test = pd.read_csv('Test_DataSet.csv').drop('ListingID', axis=1)
df_train_dummies = pd.get_dummies(df_train)
df_test_dummies = pd.get_dummies(df_test)

df_train_dummies = df_train_dummies.drop(columns=['SellerIsPriv',
       'SellerRating','SellerRevCnt','SellerZip'])
df_train_dummies = df_test_dummies.drop(columns=['SellerIsPriv',
       'SellerRating','SellerRevCnt','SellerZip'])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

df_train_dummies= At1.values.reshape(-1, 1)

X_train = df_train_dummies(['Dealer_Listing_Price'], axis=1).values
y_train = df_train_dummies['Dealer_Listing_Price'].values
X_test = df_test_dummies.values 

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(criterion='R2')
reg =RandomForestRegressor(criterion='R2', max_depth=16, n_estimators=20)
reg.fit(X_train, y_train)
reg.score(X_train, y_train)
p = reg.predict(X_test)
df_submit = pd.read_csv('Test_DataSet.csv')
df_submit['Dealer_Listing_Price'] = p
df_submit.to_csv('Test_DataSet.csv')
                                                                      
                                                                       