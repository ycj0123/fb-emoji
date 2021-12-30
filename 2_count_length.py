import pandas as pd

data = pd.read_csv('merged.csv', index_col=[0])

count = 0
for t in data['text'].to_list():
    if len(t) < 50:
        print(t)
        count += 1

print(count)