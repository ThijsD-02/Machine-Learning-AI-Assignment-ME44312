import pandas as pd
from sklearn import preprocessing

db = pd.read_csv('Ex4_NN_2025/modes.csv')
print(db.head())

X = db[['xmin', 'ymin', 'zmin', 'xmean', 'ymean', 'zmean', 'xstd', 'ystd', 'zstd']].values
print(X)
y = db['Class']
LE = preprocessing.LabelEncoder()
LE.fit(y)
y = LE.transform(y)

