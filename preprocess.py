import sys,re
import pandas as pd
import numpy  as np
import cPickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus   import stopwords
from nltk.corpus   import words
from nltk.stem     import PorterStemmer
stemmer = PorterStemmer()
number = re.compile('\d')
alpha  = re.compile('[0-9a-zA-Z]+')
def tokenise(string):
	tokens = wordpunct_tokenize(string)
	tokens = ( t for t in tokens if len(t) > 2 )
	tokens = ( t for t in tokens if alpha.match(t) )
	tokens = ( number.sub('#',t) for t in tokens )
	tokens = ( stemmer.stem(t) for t in tokens )
	return list(tokens)
#vocabulary = [ w.lower() for w in words.words('en') ]
def count(filename,field,outfile,sample=True,tokeniser=tokenise,vocab=None,chunk_out=False):
	if sample:
		chunk = 100
	elif chunk_out:
		chunk = 100000
	else:
		chunk = 100000
	data = pd.read_csv(
			filename,
			header    = 0,
			chunksize = chunk,
			index_col = 0
		)
	counter = CountVectorizer(
			tokenizer  = tokeniser,
			vocabulary = vocab,
			stop_words = stopwords.words('english'),
			min_df     = 4,
			dtype      = np.byte
		)

	if sample:
		fn = outfile + ".data"
		counts = counter.fit_transform(
			iter(chunk[field]).next() for chunk in data 
			)
		pickle.dump(counts,open(fn,'wb'),protocol=1)
	else:
		if vocab and chunk_out:
			i = 0
			for chunk in data:
				print "Making chunk %d..."%i
				counts = counter.transform(l for l in chunk[field])
				pickle.dump(counts,open("%s.%d.data"%(outfile,i),'wb'),protocol=1)
				i += 1
				exit()
		else:
			fn = outfile + ".data"
			counts = counter.fit_transform(
				l for chunk in data
				for l in chunk[field])
			print "writing %s"%fn
			pickle.dump(counts,open(fn,'wb'),protocol=1)
	if not vocab:
		pickle.dump(
				counter.vocabulary_,
				open(outfile+".vocab.data",'wb'),
				protocol=1
			)
		return counter.vocabulary_
if __name__ == "__main__":
	"""
	print "Counting Body..."
	vocab_body = count(sys.argv[1],'Body','dev.body.train')
	print "Counting Tags..."
	count(sys.argv[1],'Tags','dev.tags.train',
			tokeniser=None)
	print "Counting Test..."
	"""
	vocab_body = pickle.load(open('body.train.vocab.data'))
	count(sys.argv[2],'Body','body.test',
			vocab=vocab_body,
			sample=False,
			chunk_out=True)

