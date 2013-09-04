import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,HashingVectorizer
import sys

data = pd.read_csv(
		sys.argv[1],
		header    = 0,
		skiprows  = range(1,int(sys.argv[2])),
		nrows     = int(sys.argv[3]),
		index_col = 0
	)

counter = CountVectorizer()
counter.fit(data['Title'])
