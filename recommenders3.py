import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

food_details = pd.read_csv('final.csv')
user_ratings = pd.read_csv('user_ratings.csv', index_col=0)
X_user = user_ratings

from sklearn.decomposition import TruncatedSVD, PCA
#pca = PCA(n_components=None)titles_id.index[item_rating]
#X_user = pca.fit_transform(X_user)
#xx = pca.explained_variance_ratio_

SVD = TruncatedSVD(n_components=10)
user_based_group = SVD.fit_transform(X_user)
food_based_group = SVD.fit_transform(X_user.T)
#xx[0:11].sum()
#corr_user = np.corrcoef(user_based_group)
#corr_user = pd.DataFrame(corr_user, columns=usernames, index=usernames)

rec = np.dot(user_based_group, food_based_group.T)
corr_u = np.corrcoef(rec)
corr_u = pd.DataFrame(corr_u, columns=usernames, index=usernames)

#plt.plot(corr_u.iloc[:,0], color='blue')
'''plt.plot(corr_u.iloc[:,0],color='magenta')'''
#corr_user = np.corrcoef(user_based_group)
#corr_user = pd.DataFrame(corr_user, columns=usernames, index=usernames)

'''
plt.xlabel('Correlation Coefficients')
plt.ylabel('Amplitude')
plt.title('Comparison of coefficients')
plt.show()
'''

# First create clusters then come here!
#Similar User Choices (usernames):
user1 = 'napkinconfess'
scores = corr_u.loc[user1, :]

# Users similar to user1 based on correlation coefficients
similar_users = []
for i in list(user_ratings.index):
    if(scores[i]>0.90 and scores[i]!=1):
        similar_users.append(i)

# food rated highest by similar users are taken into consideration.
# It is highly certain that user1 will love these food rated high 
# by other similar users.
food1 = []
for each_user in similar_users:
    titles_id = user_ratings.loc[each_user]
    for item_rating in range(len(titles_id)):
        if(titles_id[item_rating] > 4.0):
            food1.append(int(titles_id.index[item_rating]))
            
'''
In the similar way, the food that is enjoyed most (>=4 rated)by the user1 is 
mapped to the cluster it belongs to and other foods of that cluster are 
recommended to the user1. It is a high chance that user1 will enjoy 
other food items of that particular cluster(s) '''

titles_id = user_ratings.loc[user1]
cluster_data = pd.read_csv('clusters.csv')
y_means = pd.read_csv('foodToCluster.csv', index_col = 0)

food2 = []
for item_rating in range(len(titles_id)):
        if(titles_id[item_rating] > 4.5):
            food2.append(int(titles_id.index[item_rating]))

food3_cluster = []
for i in range(len(food2)):
    food3_cluster.append(int(y_means['0'][food2[i]]))  

food3 = []
for i in food3_cluster:
    for eachfood in cluster_data.iloc[i]:
        if(eachfood==0):
            break
        else:
            food3.append(int(eachfood))
            
food3 = list(set(food3))
food1 = list(set(food1+food2+food3))

##Make a df containing only food1 contents 


#http://nicolas-hug.com/blog/matrix_facto_3
#https://datascienceplus.com/building-a-book-recommender-system-the-basics-knn-and-matrix-factorization/