import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus   import stopwords
from nltk.corpus   import words
import sys,re,pickle

number = re.compile('\d')
alpha  = re.compile('[0-9a-zA-Z]+')
def tokenise(string):
	tokens = wordpunct_tokenize(string)
	tokens = ( t for t in tokens if len(t) > 2 )
	tokens = ( t for t in tokens if alpha.match(t) ) 
	tokens = ( number.sub('#',t) for t in tokens )
	return list(tokens)

vocabulary = [ w.lower() for w in words.words('en') ]

if __name__ == "__main__":
	field = sys.argv[2]
	data = pd.read_csv(
			sys.argv[1],
			header    = 0,
			chunksize = 100,
			index_col = 0
		)

	counter = CountVectorizer(
			tokenizer  = tokenise,
			stop_words = stopwords.words('english'),
			min_df     = 4
		)

	counts = counter.fit_transform(
			iter(chunk[field]).next()
				for chunk in data
		)
	pickle.dump(counts,open(field.lower()+".data",'wb'))
