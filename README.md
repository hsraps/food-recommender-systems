#  Food Recommender Systems

An attempt to produce a nutrient-based meal recommender that makes meal
recommendations catering to users’ fine-grained food preferences and nutritional
needs through the use of machine learning techniques.

## Methodology

We will be trying to develop a cluster of similar food items using the dataset
consists of food items and their corresponding calorie content, protein content
and the ingredients they are made of.

Firstly, We will be importing the required Python libraries.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD, PCA
import pickle
import random
import seaborn as sns
```

```python
dataset = pd.read_csv('refined_rel.csv')
dataset.head()
```

```python
dataset.bmi[(dataset.bmi<=18.5)].hist(figsize=(10,7))
dataset.bmi[(dataset.bmi>18.5) & (dataset.bmi<=24.9)].hist(figsize=(10,7))
dataset.bmi[(dataset.bmi>24.9) & (dataset.bmi<40)].hist(figsize=(10,7))
```

![image](https://github.com/hsraps/food-recommender-systems/blob/master/img1.png)

Less than 18.5 > Undernourished (Requires medical guidance)

18.5 to 24.9 >
Normal BMI

greater 24.9 > Overnourished

```python
X = dataset.iloc[:,[1,2,4,5]].values
y = dataset.iloc[:, 6].values
X
```

```python
ohe1 = OneHotEncoder(categorical_features = [3])
X = ohe1.fit_transform(X).toarray()
X = X[:,1:]
X
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =0)
X_train.shape
```

```python
X_test.shape
```

```python
regressor = RandomForestRegressor(n_estimators = 20, random_state=0)
regressor.fit(X_train ,y_train)
y_pred = regressor.predict(X_test)
```

```python
scores = cross_val_score(estimator = regressor, X = X, y = y, cv = 10)
scores.mean()
```

```python
scores.std()
```

This means our predicted daily calorie intake would be correct in 95% of the
cases.

Our Pre-trained model can be saved as follows:

```python
save_reg = open('regressor.pickle','wb')
pickle.dump(regressor, save_reg)
save_reg.close()

'''
#Loading a regressor
f = open('regressor.pickle','rb')
r2 = pickle.load(f)
f.close()
user_calories = r2.predict(X_user)
'''
```

```python
user_ratings = pd.read_csv('user_ratings.csv', index_col=0)
user_ratings.head()
```

This sparse matrix contains the record of users(rows) of how much they have
liked a food-item(columns) on a scale of 1-5.
The user have not rated all the
food they ate, which are represented by 0 (means NaN)

```python
SVD = TruncatedSVD(n_components=10)
user_based_group = SVD.fit_transform(user_ratings)
user_based_group.shape
```

```python
food_based_group = SVD.fit_transform(user_ratings.T)
food_based_group.shape
```

```python
corr_user = np.corrcoef(user_based_group)
corr_user
```

We are concerned with knowing the similar user taste so that we can recommend
what other user enjoys based on their ratings. So, we find the **Pearson
Correlation Coefficient** for the same

```python
_, ax = plt.subplots(figsize=(15,15))
sns.heatmap(corr_user, xticklabels= user_ratings.index, yticklabels = user_ratings.index, vmin=0.90, vmax=0.999, ax=ax)
```
![image](https://github.com/hsraps/food-recommender-systems/blob/master/img2.png)

We will only be considering the coefficient values above 0.9 for each user, as
can clearly be seen in the above heatmap.

```python
food_details = pd.read_csv('final.csv')
food_details.head()
```

```python
X_food = food_details.iloc[:, 1:]
X_food = X_food.drop(['calories','protein','fat','categories','sodium','title'],axis=1) 
X_food = X_food.set_index(food_details.foodID)
X
```
![alt text](https://raw.githubusercontent.com/hsraps/xx/branch/path/to/img.png)

```python
SVD = TruncatedSVD(n_components=60, random_state=0)
new_X = SVD.fit_transform(X_food)
new_X = pd.DataFrame(new_X, index = food_details.foodID)
new_X.head()
```

```python
import scipy.cluster.hierarchy as sch
plt.subplots(figsize=(20,7))
plt.axhline(y=24)
plt.ylabel("No. of Clusters")
plt.xlabel("Features (ingredients)")
dendrogram = sch.dendrogram(sch.linkage(new_X, method="ward"))
```
![image](https://github.com/hsraps/food-recommender-systems/blob/master/img3.png)

We will be considering K=24 clusters to put similar food into the same category.
Apply K-Means Clustering afterwards

```python
kmeans = KMeans(n_clusters = 24)
y_means = kmeans.fit_predict(new_X)
```

```python
y_means
```

Through the use of above model, using user's BMI we can predict his daily
calorie requirement using **Calorie prediction model**. Ask for user's feedback
about different food items and store as ratings. Use the cluster of similar food
items that we created above. Recommend other food items belonging to that
cluster and also similar user taste preferences.

This will generate a corresponding food dataset for each user. This dataset can
then be processed to extract food as per need for breakfast, lunch or dinner.
The sum of calorie consumed per day should satisfy daily calorie requirement.

In this way Clustering algorithms and Dimensionality reduction techniques can be
used to produce a simple food recommender systems. There is always a scope for
improving the efficiency and accuracy of the model.
