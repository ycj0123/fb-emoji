import pandas as pd

data = pd.read_csv('clean_corpus/fb6000.csv', index_col=[0])

rtns = { 'like': [], 'love': [], 'care': [], 'haha': [], 'wow': [], 'sad': [], 'angry': [] }
for c in data.columns:
    if c == 'text' or c == 'page' or c == 'id':
        continue
    rtns[c] = sum(data[c].to_list())

total = sum(rtns.values())
for k in rtns:
    rtns[k] = rtns[k]/total
print(rtns)