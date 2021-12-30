#kmeans

from ckip_transformers.nlp import CkipWordSegmenter
import pandas as pd
import re
import emoji

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

# load and clean text
data = pd.read_csv('merged.csv', index_col=[0])
data = data.drop([495,943,1726,1769])
corpus = data['text'].tolist()
for i in range(len(corpus)):
    corpus[i] = emoji.demojize(corpus[i])
text_cleaning_re = '[a-zA-Z0-9~!@#$%^&*()_+{}|:\"\<\>?\[\]\\;,./]'
text_keep_re = ':.+:|[\u4E00-\u9FD5]'
corpus_clean = keep(corpus, text_keep_re)
for i in range(len(corpus_clean)):
    corpus_clean[i] = emoji.emojize(corpus_clean[i])
    corpus_clean[i] = corpus_clean[i].replace(" ", "")
corpus_clean = clean(corpus_clean, text_cleaning_re)
data['text'] = corpus_clean
data = data.replace(r'^s*$', float('NaN'), regex = True)
data.dropna(inplace = True)
corpus_clean = [l + '\n' for l in corpus_clean]
data.to_csv('clean.csv', index=False)
data = pd.read_csv('clean.csv')
data.to_csv('clean.csv', index=True)