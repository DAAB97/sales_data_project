# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 03:33:03 2020

@author: abderrahman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/abdo/Desktop/sales_data.csv')

#cleanning data 

## detect unique values in the columns
for cat in df.columns:
    print(cat a, df[cat].unique())
    
## replacing all unknown data from gender, age and child with null  
df['gender'] = df.gender.replace('U', np.NaN)
df['age'] = df.age.replace('1_Unk', np.NaN)
df['child'] = df.child.replace('U', np.NaN)
df['child'] = df.child.replace('0', np.NaN)

##calculate the null valuesin each column
df.isnull().sum()


##Returns stacked bar plot
def category_stackedbar(df, category):
    return pd.DataFrame(df.groupby(category).count()['flag'] / df.groupby(category).count()['flag'].sum() * 100).rename(columns={"flag": "%"}).T.plot(kind='bar', stacked=True);


category_stackedbar(df, 'house_owner');
df['house_owner'] = df['house_owner'].fillna(df.mode()['house_owner'][0])


for cat in df.columns:
    print(cat, df[cat].unique())


df.isnull().sum()


category_stackedbar(df, 'age');

## drop column
df = df.dropna(subset=['age'])


category_stackedbar(df, 'child');
df = df.drop('child', axis=1)

category_stackedbar(df, 'marriage');

## replace null values with the mode
df['marriage'] = df['marriage'].fillna(df.mode()['marriage'][0])

##drop 2 columns
df = df.dropna(subset=['gender', 'education'])

df.isnull().sum()


## change specific values in columns 
df['flag'] = df['flag'].apply(lambda value: 1 if value == 'Y' else 0)
df['online'] = df['online'].apply(lambda value: 1 if value == 'Y' else 0)
df['education'] = df['education'].apply(lambda value: int(value[0]) + 1)
df['age'] = df['age'].apply(lambda value: int(value[0]) - 1)
df['mortgage'] = df['mortgage'].apply(lambda value: int(value[0]))


## change fam_income into numbers 
dict_fam_income_label = {}
for i, char in enumerate(sorted(df['fam_income'].unique().tolist())):
    dict_fam_income_label[char] = i + 1

df['fam_income'] = df['fam_income'].apply(lambda value: dict_fam_income_label[value])


for cat in df.columns:
    print(cat, df[cat].unique())

    
    

    
###############################################################define the model data frame     
df.columns

df_model = df[['flag', 'gender', 'education', 'house_val', 'age', 'online', 'marriage', 'occupation', 'mortgage', 'house_owner',
       'region', 'fam_income']]

df_dum = pd.get_dummies(df_model)


df_dum.head()

###############################################################


from sklearn.model_selection import train_test_split

X = df_dum.drop('flag', axis = 1)
y = df_dum.flag.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

# LinearRegression model
lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))


#Lasso model
lm_l = Lasso(alpha=0.01)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lm_l = Lasso(alpha=i/100)
    error.append(np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))
    
plt.plot(alpha, error)

err = tuple(zip(alpha, error))
err

# random forest 
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

#np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

## improve the random forest model with GridSearchCV 
from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators':range(10, 300, 10), 'criterion': ('mse', 'mae'), 'max_features': ('auto', 'sqrt', 'log2')}
gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)
gs.best_score_



########################### Productionize the model with Flask 
import pickle
pickl = {'model': rf}
pickle.dump(pickl, open( 'model_file' + ".p", "wb" ))


file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']


    
# model.predict(X_test.iloc[1, :].values.reshape(1, -1))
# X_test.iloc[1, :].values

























