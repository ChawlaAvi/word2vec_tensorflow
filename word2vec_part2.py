import pickle
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np



with open('word_vecs.pickle','rb') as h:
	vectors = pickle.load(h)

with open('word2int.pickle','rb') as h:
	word2int = pickle.load(h)

with open('int2word.pickle','rb') as h:
	int2word = pickle.load(h)

with open('words.pickle','rb') as h:
	all_words = pickle.load(h)	


model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors)	

normalizer = preprocessing.Normalizer()
vectors =  normalizer.fit_transform(vectors, 'l2')


fig, ax = plt.subplots()
for word in all_words:
    
    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))

plt.show()


