import numpy as np
from numpy.ma import size
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import emoji
from random import sample
from tools import text_has_emoji
w2v_path = 'w2v_5.model'
n_text = 150

w2v_model = Word2Vec.load(w2v_path)
vocab = list(w2v_model.wv.key_to_index.keys())
vocab_emoji = [v for v in vocab if text_has_emoji(v)][:120]
vocab_text = list(set(vocab) - set(vocab_emoji))
vocab_sample = sample(vocab_text,k=n_text) + vocab_emoji
# vocab_sample = vocab
# vocab_sample = vocab_emoji
# demoj_vocab = [emoji.demojize(v) for v in vocab]
print(len(vocab_emoji))
print(vocab_sample)
X = w2v_model.wv[vocab_sample]

embed = TSNE(n_components=2, learning_rate='auto', init='pca', verbose=1).fit_transform(X)
# print(embed)
embed = embed.transpose()

plt.figure(figsize=(9, 9))
plt.scatter(embed[0],embed[1])
simhei = FontProperties(fname='SimHei.ttf', size='xx-large')
acemoji = FontProperties(fname='Apple Color Emoji.ttc', size='xx-large')
for i, label in enumerate(vocab_sample):
    if i < n_text:
        plt.annotate(label, (embed[0][i], embed[1][i]), FontProperties = simhei)
    else:
        plt.annotate(label, (embed[0][i], embed[1][i]), FontProperties = acemoji)

plt.savefig("viz.png")