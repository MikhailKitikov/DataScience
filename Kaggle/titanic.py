import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# dealing with cathegorical values
def one_hot_encode(train, test):
    one_hot_encoded_training_predictors = pd.get_dummies(train)
    one_hot_encoded_test_predictors = pd.get_dummies(test)
    train, test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join = 'left', axis = 1)
    return  train, test
        
# finding a substring
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    return np.nan
    
def first_phase_clean(data):
    
    # assigning nan cabins to 'Unknown'
    data.Cabin = data.Cabin.fillna('Unknown') 
    
    # getting titles    
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev', 'Dr', 'Ms', 'Mlle','Col', 
                    'Capt', 'Mme', 'Countess', 'Don', 'Jonkheer']
    data['Title'] = list(map(lambda x: substrings_in_string(x, title_list), data.Name))

    #replacing all titles with mr, mrs, miss, master
    def replace_titles(x):
        title = x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title 

    data['Title'] = data.apply(replace_titles, axis = 1)
    
    
    data["FamilyName"]=data["Name"].map(lambda x: x.split(",")[0].strip())
    data["TwoLetters"]=data["FamilyName"].map(lambda x: x[-2:])
    data.drop(['Name'], axis = 1)
    
    # getting decks
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    data['Deck'] = list(map(lambda x: substrings_in_string(str(x), cabin_list), data['Cabin']))
    data.drop(['Cabin'], axis = 1)
    
    # considering family
    data['Family_Size'] = data['SibSp'] + data['Parch']   
    
    # fare per person
    data['Fare_Per_Person'] = data['Fare'] / (data['Family_Size'] + 1)
    data.drop(['Fare'], axis = 1)
    
    data['n'] = data['Age'] * data['Pclass']
    return data

def last_phase_clean(train, test, train_y, test_y):
    # mothers and sisters

    dfFamilyTrain = train[(train["Parch"]>0)&(train_y==0)]
    dfFamily = dfFamilyTrain[(dfFamilyTrain["Sex"]=="female")|(dfFamilyTrain["Age"]<10)]
    familiestrain = dfFamily["FamilyName"]
    
    dfFamilytest = test[(test["Parch"]>0)&(test["Sex"]=="female")]
    familiestest = dfFamilytest["FamilyName"]
    intersection = np.intersect1d(familiestest,familiestrain)
    
    test_y[test.PassengerId.isin(intersection)] = 0
    
    return test_y

def second_phase_clean(train, test):
    # handling nan values
    my_imputer = Imputer()
    train = pd.DataFrame(my_imputer.fit_transform(train), index = train.index, columns = train.columns)
    test = pd.DataFrame(my_imputer.transform(test), index = test.index, columns = test.columns)

    return train, test

# data reading
file_path = '../input/train.csv'
home_data = pd.read_csv(file_path)
test_data_path = '../input/test.csv'
test_data = pd.read_csv(test_data_path)

# feature processing
train_y = home_data.Survived
train_x = home_data.drop(['Survived'], axis = 1)
test_data = test_data

# 1st phase
train_x = first_phase_clean(train_x)
test_data = first_phase_clean(test_data)

ptrain_x, ptest_data = train_x, test_data

# encoding
train_x, test_data = one_hot_encode(train_x, test_data)

# 2nd phase
train_x, test_data = second_phase_clean(train_x, test_data)

# model building
model = XGBRegressor()
model.fit(train_x, train_y)
predictions = model.predict(test_data)

# last phase
predictions = last_phase_clean(ptrain_x, ptest_data, train_y, predictions)

# outputting
pd.DataFrame({'PassengerId': test_data.PassengerId.astype(int), 'Survived': [int(p > 0.5) for p in predictions]}).to_csv('submission.csv', index = False)