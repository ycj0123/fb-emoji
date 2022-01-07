#kmeans

from ckip_transformers.nlp import CkipWordSegmenter
import pandas as pd
import re
import emoji

input_file = 'original_corpus/ig5000.csv'
output_file = 'data_ig.csv'

def keep(texts, keep_re):
    output = []
    for text in texts:
        text = re.findall(keep_re, text)
        text = ' '.join(text)
        output.append(text)
    return output

def clean(texts, clean_re):
    output = []
    for text in texts:
        text = re.sub(clean_re, '', text)
        output.append(text)
    return output

# load and clean specified rows
data = pd.read_csv(input_file)
# data = data.drop([1995,2515])
# data = data.drop(range(2017,2074))

# drop posts without text and demojize
data = data.replace(r'^s*$', float('NaN'), regex = True)
data.dropna(inplace = True)
corpus = data['text'].tolist()
for i in range(len(corpus)):
    corpus[i] = emoji.demojize(corpus[i])

# clean text and re-emojize
text_cleaning_re = '[a-zA-Z0-9~!@#$%^&*()_+{}|:\"\<\>?\[\]\\;,./]'
text_keep_re = ':.+:|[\u4E00-\u9FD5]'
corpus_clean = keep(corpus, text_keep_re)
for i in range(len(corpus_clean)):
    corpus_clean[i] = emoji.emojize(corpus_clean[i])
    corpus_clean[i] = corpus_clean[i].replace(" ", "")
corpus_clean = clean(corpus_clean, text_cleaning_re)

# drop posts without text again
data['text'] = corpus_clean
data = data.replace(r'^s*$', float('NaN'), regex = True)
data.dropna(inplace = True)

# drop rows with short text and few reactions
new_data = data.copy()
for i in data.index:
    if len(data['text'][i]) < 4:
        # print(data['text'][i])
        new_data = new_data.drop(i)

# handle the index problem
new_data.to_csv(output_file, index=False)
new_data = pd.read_csv(output_file)
new_data.to_csv(output_file, index=True)