from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# set np print options
np.set_printoptions(precision=3, suppress=True)

# read data and normalize different reactions
data = pd.read_csv('clean.csv', index_col=[0])
text = data['text'].tolist()
reactions = data.drop('text', axis=1).to_numpy()
# weights = np.sum(reactions, axis=0)
# weights = np.max(weights)/weights
weights = np.ones(reactions.shape)
weights[:, 1:] = 10
print(weights)
reactions = np.multiply(reactions,weights)

# calculate reactions' proportion
re_sum = np.repeat(np.expand_dims(np.sum(reactions, axis=1), axis=1), 7, axis=1)
re_sum[np.where(re_sum==0)] = 1
reactions = np.divide(reactions, re_sum)
print(reactions.shape)
mask = np.all(np.isnan(reactions) | np.equal(reactions, 0), axis=1)
reactions = reactions[~mask]
print(reactions.shape)

# k means cluastering
kms = KMeans(n_clusters=6).fit(reactions)
print(kms.cluster_centers_)
# preds = kms.predict(reactions)
# print(preds)
# _, count = np.unique(preds, return_counts=True)
# print(count)
# centers = pd.DataFrame(np.around(kms.cluster_centers_, decimals=3))
# centers.to_csv('centers.csv')

# labels = pd.DataFrame(list(zip(text,preds)), columns=['text', 'label'])
# labels.to_csv('labels.csv')
print(np.argmax(kms.cluster_centers_, axis=1))
