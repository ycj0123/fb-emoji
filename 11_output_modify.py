# import matplotlib, mplcairo
# print(matplotlib.matplotlib_fname())
# print('Default backend: ' + matplotlib.get_backend()) 

import pandas as pd

changed_csv = 'ig_changed.csv'
unchanged_csv = 'ig_unchanged.csv'

# ig = pd.read_csv('clean_corpus/ig5000.csv', index_col=[0])

# ig_nonad = ig.copy()
# for i, row in ig.iterrows():
#     if int(row['label'])==1:
#         ig_nonad = ig_nonad.drop(i)
# ig_nonad.to_csv('ig_nonad.csv', index=False)
# ig_nonad = pd.read_csv('ig_nonad.csv')
# ig_nonad.to_csv('ig_nonad.csv')

ig = pd.read_csv('output.csv', index_col=[0])

ig_changed = ig.copy()
for i, row in ig.iterrows():
    if int(row['emoji label']) ==  int(row['emoji-less label']):
        ig_changed = ig_changed.drop(i)
ig_changed.to_csv(changed_csv, index=False)
ig_changed = pd.read_csv(changed_csv)
ig_changed.to_csv(changed_csv)

ig_unchanged = ig.copy()
for i, row in ig.iterrows():
    if int(row['emoji label']) !=  int(row['emoji-less label']):
        ig_unchanged = ig_unchanged.drop(i)
ig_unchanged.to_csv(unchanged_csv, index=False)
ig_unchanged = pd.read_csv(unchanged_csv)

diff = []
for i, row in ig_unchanged.iterrows():
    diff.append(row['emoji std'] - row['emoji-less std'])

ig_unchanged['difference'] = diff
ig_unchanged.to_csv(unchanged_csv)