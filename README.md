# diet-recommender-systems

## Calories Prediction based on Body Index
The calories requirement for each day is predicted on the basis of user’s body index through the use of Random Forest Regression Technique, using sklearn library. The  dataset contains dummy variables which are manipulated using OneHotEncoder.

## Recommender Systems Techniques Used
The webapp uses **Collaborative filtering methods** to recommend food to the user. The food recommended is healthy, nutritious and it will satisfy user’s taste. The WebApp uses two algorithms: 

* TruncatedSVD is applied on sparse user ratings and is decomposed to extract *typical latent factors*. The Pearson moment correlation coefficient(PMCC) is then generated for each users to get correlations between users. This data takes care of user’s taste buds.
The user-ratings data stores the user ratings for each user for each food Items. It is very unlikely that user rates all 5000 dishes, so the sparse user ratings matrix is obtained. PMCC helps to know similar user tastes based on user ratings.

* The dimensions of the Count-vectorised features are reduced to 60 to increase computation time using PCA and SVD techniques called **Latent Semantic Analysis**. KMeans clustering is then applied over the dataset and clusters are generated based on similar tags/ingredients associated with each food item. Elbow Method is used to know optimum number of clusters required.

Through the use of **KMeans Clustering** algorithms, 250 Clusters are maintained for **5000 food items** datasets. Through the use of Elbow Method. Food items are clustered based on their ingredients. The food that is enjoyed most (>=4 rated)by the user1 is mapped to the cluster it belongs to and other foods of that cluster are recommended to the user1. It is a high chance that user1 will enjoy other food items of that particular cluster(s).

The dataset obtained (recommended ) for each user is then generated. The calories required on daily basis is then divided into three meals stated- Breakfast, Lunch, Dinner. The Meals combinations are then suggested from the available food dataset for a user.  

