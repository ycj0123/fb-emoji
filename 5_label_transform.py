import pandas as pd
import numpy as np

data = pd.read_csv('labels.csv', index_col=[0])
text = data['text'].tolist()
labels = data['label'].to_numpy()
labels_new = np.copy(labels)
labels_new[np.where(labels==0)] = 3
labels_new[np.where(labels==1)] = 6
labels_new[np.where(labels==2)] = 0
labels_new[np.where(labels==3)] = 1
labels_new[np.where(labels==4)] = 4
labels_new[np.where(labels==5)] = 5
labels_new[np.where(labels==6)] = 2

labels = pd.DataFrame(list(zip(text,labels_new)), columns=['text', 'label'])
labels.to_csv('labels_new.csv')