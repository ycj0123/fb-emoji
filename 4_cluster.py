from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import pickle

num2rtn = {0: 'like', 1: 'love', 2: 'care', 3: 'haha', 4: 'wow', 5: 'sad', 6: 'angry'}
# set np print options
np.set_printoptions(precision=3, suppress=True)

# read data and normalize different reactions
data = pd.read_csv('clean_corpus/fb6000.csv', index_col=[0])
# text = data['text'].tolist()
reactions = data.drop(['page', 'id', 'text'], axis=1).to_numpy()
# weights = np.sum(reactions, axis=0)
# weights = np.max(weights)/weights
weights = np.ones(reactions.shape)
weights[:, 1:] = 10
# print(weights)
# print(reactions)
reactions = np.multiply(reactions,weights)

# calculate reactions' proportion
re_sum = np.repeat(np.expand_dims(np.sum(reactions, axis=1), axis=1), 7, axis=1)
re_sum[np.where(re_sum==0)] = 1
reactions = np.divide(reactions, re_sum)
# print(reactions.shape)
mask = np.all(np.isnan(reactions) | np.equal(reactions, 0), axis=1)
reactions = reactions[~mask]
# print(reactions.shape)

# k means cluastering
kms = KMeans(n_clusters=5).fit(reactions)
# print(kms.cluster_centers_)
centers = pd.DataFrame(kms.cluster_centers_, columns=['like', 'love', 'care', 'haha', 'wow', 'sad', 'angry'])
print(centers)

for i, r in enumerate(np.argmax(kms.cluster_centers_, axis=1)):
    print(f'{i}: {num2rtn[r]}', end=' ')
print()

with open("kmeans.pkl", "wb") as f:
    pickle.dump(kms, f)