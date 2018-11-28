import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('refined_rel.csv')

X = dataset.iloc[:,[1,2,4,5]].values
y = dataset.iloc[:, 6].values

'''
for alg.py
Take user I/P :
    height(in m), weight(in kgs), active state (0-sedentary,1-active, 2-highly active)
'''

user_height = 1.71
user_weight = 56
user_active = 1

user_bmi = user_weight/(user_height**2)
X_user = np.array([user_height, user_weight, user_bmi, user_active]).reshape(1,-1)

from sklearn.preprocessing import OneHotEncoder
ohe1 = OneHotEncoder(categorical_features = [3])
X = ohe1.fit_transform(X).toarray()
X_user = ohe1.transform(X_user).toarray() #For user
X = X[:,1:]
X_user = X_user[:,1:] #For user

'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 20, random_state=0)
regressor.fit(X_train ,y_train)
y_pred = regressor.predict(X_test)

#F_FOld cross validation is applied
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator = regressor, X = X, y = y, cv = 10)
scores.mean()
scores.std()

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 20, random_state=0)
regressor.fit(X ,y)

#Saving the regressor
import pickle
save_reg = open('regressor.pickle','wb')
pickle.dump(regressor, save_reg)
save_reg.close()
'''
#Loading a regressor that is trained and saved for original dataset named regressor.pickle
f = open('regressor.pickle','rb')
r2 = pickle.load(f)
f.close()
user_calories = r2.predict(X_user)