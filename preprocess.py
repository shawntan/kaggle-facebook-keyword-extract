import pandas as pd
import sys
data = pd.read_csv(
		sys.argv[1],
		header   = 0,
		skiprows = range(1,int(sys.argv[2])),
		nrows    = int(sys.argv[3])
	)
