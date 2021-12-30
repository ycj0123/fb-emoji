from ckip_transformers.nlp import CkipWordSegmenter
import pandas as pd
import fasttext

def tokenize(texts, driver, batch_size, max_length):
    output=[]
    ws  = driver(texts, use_delim=False, batch_size = batch_size, max_length = max_length)
    output = [" ".join(ls) for ls in ws]
    return output

data = pd.read_csv('clean.csv', index_col=[0])
corpus = data['text'].tolist()

# tokenize
ws_driver = CkipWordSegmenter(device=0, level=3)
corpus_segmented = tokenize(corpus, ws_driver, 256, 300)

# train embedding
model = fasttext.train_unsupervised('clean.csv', dim=300, minCount=3, model='skipgram', wordNgrams=5, neg=10)
