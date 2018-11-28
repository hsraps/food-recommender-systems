import numpy as np
import pandas as pd
import random

user1 = 'shivanshu'
food_dataset = pd.read_csv('final.csv')
ratings_dataset = pd.read_csv('user_ratings.csv', index_col=0)
rate_this_food = {}

ids = []

for i in range(10):
    index = np.random.randint(0,5000)
    rate_this_food[food_dataset.foodID[index]] = food_dataset.title[index]
    ids.append(food_dataset.foodID[index])
ids.sort()

'''fetched from user
get a dict foodId: rating'''
##this part get from webapp--------------
for i in range(10):
    rate_this_food[ids[i]] = float(np.random.randint(1,6))rate_this_food[ids[i]] = float(np.random.randint(1,6))
    
    
    rate_this_food[ids[0]] = 5
    rate_this_food[ids[1]] = 5
    rate_this_food[ids[2]] = 5
    rate_this_food[ids[3]] = 5
    rate_this_food[ids[4]] = 5
    rate_this_food[ids[5]] = 1
    rate_this_food[ids[6]] = 1
    rate_this_food[ids[7]] = 5
    rate_this_food[ids[8]] = 4
    rate_this_food[ids[9]] = 3
###get from webapp----------------------
    
sparse_ratings = pd.Series(np.zeros(5000).astype('float'), index= food_dataset['foodID'])
for i in range(10):
    sparse_ratings[ids[i]] = float(rate_this_food[ids[i]])
sparse_ratings=list(sparse_ratings)

ratings_dataset.loc[user1] = sparse_ratings
ratings_dataset.to_csv('user_ratings.csv')

'''proceed to clustering.py'''