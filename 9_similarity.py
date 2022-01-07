import emoji
from gensim.models import Word2Vec
from matplotlib.pyplot import axis
from tools import text_has_emoji
import numpy as np
import pandas as pd

def cosine_similarity(x,y):
    num = x.dot(y)
    num = num.squeeze(1)
    # print(np.linalg.norm(x, axis=1).shape, np.linalg.norm(y))
    denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y)
    # print(num.shape, denom.shape)
    return np.divide(num, denom)

w2v_path = 'w2v_10_test.model'
w2v_model = Word2Vec.load(w2v_path)

w2v_model = Word2Vec.load(w2v_path)
vocab = np.array(list(w2v_model.wv.key_to_index.keys()))
vocab_emoji = np.array([v for v in vocab if text_has_emoji(v)])
vocab_text = np.array(list(set(vocab) - set(vocab_emoji)))

all_emb = w2v_model.wv[vocab]
emoji_emb = w2v_model.wv[vocab_emoji]
# print(vocab_emoji[0], emoji_emb[0])
# print(w2v_model.wv[vocab_emoji[0]])
# quit()
text_emb = w2v_model.wv[vocab_text]
n_text = len(vocab_text)
# embs are np arrays

# ed = []
# # Euclidean distance
# for i, e in enumerate(vocab_emoji):
#     emb = emoji_emb[i]
#     emb = np.broadcast_to(emb,(n_text, len(emb)))
#     dist = np.sum(np.square(text_emb-emb), axis=1)
#     idx = np.argsort(dist)[:4]
#     matches = vocab_text[idx]
#     ed.append(matches)
#     # print(e, matches)

cs = []
# cosine similarity
for i, e in enumerate(vocab_emoji):
    emb = emoji_emb[i]
    emb = np.expand_dims(emb, 1)
    # print(text_emb.shape, emb.shape)
    sim = cosine_similarity(all_emb, emb)
    idx = np.argsort(sim)[-6:-1]
    idx = np.flip(idx)
    # print(sim.shape)
    # print(sim[idx], sim[:10])
    matches = vocab[idx]
    print(e, matches)
    cs.append(matches)

cs = np.array(cs).T
output = pd.DataFrame(list(zip(vocab_emoji, *cs)))
output.to_csv('emoji_text.csv')
