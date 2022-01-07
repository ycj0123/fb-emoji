import pandas as pd
import numpy as np

data = pd.read_csv('labels.csv', index_col=[0])

reactions = [0,0,0,0,0]

for i in data['label']:
    reactions[int(i)] += 1

print(reactions)