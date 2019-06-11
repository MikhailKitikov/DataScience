import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
    
# main

file_path = '../input/train.csv'
home_data = pd.read_csv(file_path)
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)

y = home_data.SalePrice
train = home_data.drop(['SalePrice', 'EnclosedPorch', 'LowQualFinSF', 'MiscVal', 'OpenPorchSF', 'PoolArea', 'ScreenPorch'], axis = 1)
test = test_data.drop(['EnclosedPorch', 'LowQualFinSF', 'MiscVal', 'OpenPorchSF', 'PoolArea', 'ScreenPorch'], axis = 1)

one_hot_encoded_training_predictors = pd.get_dummies(train)
one_hot_encoded_test_predictors = pd.get_dummies(test)
train, test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join = 'left', axis = 1)

my_imputer = Imputer()
train = my_imputer.fit_transform(train)
test = my_imputer.transform(test)

model = XGBRegressor()
model.fit(train, y)
preds = model.predict(test)

# outputting

pd.DataFrame({'Id': test_data.Id, 'SalePrice': preds}).to_csv('submission.csv', index = False)