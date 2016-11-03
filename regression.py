"""Linear regression tutorial.

Tutorial: Regression Intro - Practical Machine Learning Tutorial with
Python p.2 by sentdex
"""

import quandl
import math
import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import pickle
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression


style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Volume', 'Adj. Close']]
df['hi_lo_percent'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['percent_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'hi_lo_percent', 'percent_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)     # Fill missing data, treat as outlier
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
X = np.array(df.drop(['label'], 1))     # features, drop labels
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])   # labels

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.2)

# enable to re-train
# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)
# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

with open('linearregression.pickle', 'rb') as pickle_in:
    clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

print('accuracy:', accuracy)
print('forecast set:', forecast_set)
print('forecast days out:', forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
