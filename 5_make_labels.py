import pickle
import numpy as np
import pandas as pd

with open("kmeans.pkl", "rb") as f:
    kms = pickle.load(f)

# read data and normalize different reactions
data = pd.read_csv('clean_corpus/fb6000.csv', index_col=[0])
text = data['text'].tolist()
reactions = data.drop(['page', 'id', 'text'], axis=1).to_numpy()
weights = np.ones(reactions.shape)
weights[:, 1:] = 10
reactions = np.multiply(reactions,weights)

# calculate reactions' proportion
re_sum = np.repeat(np.expand_dims(np.sum(reactions, axis=1), axis=1), 7, axis=1)
re_sum[np.where(re_sum==0)] = 1
reactions = np.divide(reactions, re_sum)

preds = kms.predict(reactions)
# print(preds)
# _, count = np.unique(preds, return_counts=True)
# print(count)
centers = pd.DataFrame(np.around(kms.cluster_centers_, decimals=3), columns=['like', 'love', 'care', 'haha', 'wow', 'sad', 'angry'])
centers.to_csv('centers.csv')

labels = pd.DataFrame(list(zip(text,preds)), columns=['text', 'label'])
labels.to_csv('labels.csv')

num2rtn = {1: 'like', 2: 'haha', 0: 'wow', 3: 'love', 4: 'care/sad/angry'}