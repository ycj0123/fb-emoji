from ckip_transformers.nlp import CkipWordSegmenter
from gensim.models import Word2Vec
import pandas as pd
# import fasttext
import torch

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
# corpus_segmented = []
# for post in corpus:
#     corpus_segmented.append(' '.join(post))

# write to txt
# corpus_segmented = [i + '\n' for i in corpus_segmented]
# with open('seg.txt', 'w') as f:
#     f.writelines(corpus_segmented)

# train embedding
w2v_corpus = [text.split() for text in corpus_segmented]
print(w2v_corpus[:5])
w2v_model = Word2Vec(size = 300, window = 5, min_count = 1, workers = 8, batch_words = 10000) #sg = 1 : use skip-gram model
w2v_model.build_vocab(w2v_corpus)
w2v_model.train(w2v_corpus, total_examples=len(w2v_corpus), epochs = 32)

words_w2v = []
for word in w2v_corpus[87]:
    if word in w2v_model.wv.vocab and not word in words_w2v:
        words_w2v.append(word)
w2v_d = pd.DataFrame(w2v_model.wv[words_w2v], index=words_w2v)
w2v_d.to_csv('embedding.csv')