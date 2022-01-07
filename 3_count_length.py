import pandas as pd

data = pd.read_csv('clean_corpus/ig5000.csv', index_col=[0])

count = 0
for t in data['text'].to_list():
    count += len(t)

print(count)