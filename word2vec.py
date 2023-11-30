from nltk.corpus import brown
import tensorflow as tf
from nltk.corpus import stopwords
import numpy as np
import pickle

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

f=open('data.txt','r')

r = [((((l.strip()).lower()).split('.'))[0]).split() for l in f]
# print(r)

all_words = []

for i in r:
    all_words.extend(iter(i))
all_words = set(all_words)

word2int = {}
int2word = {}
vocab_size = len(all_words)

for i,word in enumerate(all_words):
    word2int[word] = i
    int2word[i] = word


data = []
WINDOW_SIZE = 3
for sentence in r:
    for word_index, word in enumerate(sentence):
        data.extend(
            [word, nb_word]
            for nb_word in sentence[
                max(word_index - WINDOW_SIZE, 0) : min(
                    word_index + WINDOW_SIZE, len(sentence)
                )
                + 1
            ]
            if nb_word != word
        )
# print(data)

x_train = []
y_train = [] 

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))


x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# print(x_train)
# print(x_train.shape, y_train.shape)

x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 8
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))
hidden_representation = tf.add(tf.matmul(x,W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) 

cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))


train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 20000

for i in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print(i , n_iters , ' loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))


vectors = sess.run(W1 + b1)

sess.close()

with open('word_vecs.pickle','wb') as h:
	pickle.dump(vectors,h)

with open('word2int.pickle','wb') as h:
	pickle.dump(word2int,h)

with open('int2word.pickle','wb') as h:
	pickle.dump(int2word,h)

with open('words.pickle','wb') as h:
	pickle.dump(all_words,h)	