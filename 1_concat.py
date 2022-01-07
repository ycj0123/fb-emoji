import sys
import pandas as pd

files = []
for i in range(1, len(sys.argv)):
    files.append(pd.read_csv(sys.argv[i], index_col=[0]).dropna())

output = pd.concat(files, ignore_index=True)
# output = output.drop(columns=[output.columns[0]])
output.to_csv('merged.csv')