import cPickle as pickle
from scipy.sparse import coo_matrix
from itertools import izip
model = pickle.load(open('model.data'))
X     = pickle.load(open('body.test.0.data'))
vocab = pickle.load(open('tags.train.vocab.data'))
vocab_list = [None] * len(vocab)
for k,i in vocab.iteritems(): vocab_list[i] = k
vocab = vocab_list
print "Files loaded.."
Y = model.predict(X)
print "predicted.."
for row in Y:
	print ' '.join(vocab[c] for c in row)
