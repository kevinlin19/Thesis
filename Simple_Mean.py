import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import check_array

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Use of this metric is not recommended; for illustration only.
    See other regression metrics on sklearn docs:
      http://scikit-learn.org/stable/modules/classes.html#regression-metrics
    Use like any other metric
    >>> y_true = [3, -0.5, 2, 7]; y_pred = [2.5, -0.3, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    Out[]: 24.791666666666668
    """
    y_true = check_array(y_true)
    y_pred = check_array(y_pred)
    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


df = pd.read_csv('./week.csv', names=['Date', 'Counts']) # (144, 2)
X = df[:-8] # 136
Y = df[15:] # 129
X_train = X[:int(X.shape[0]*0.8)] # 108
X_test = X[int(X.shape[0]*0.8)-1:] # 29
X['Simple_mean'] = np.array([df['Counts'][:i].mean() for i in range(len(df))])[:-8].reshape(-1, 1)
test = X_test['Counts'].reshape(1, -1)
pred = X['Simple_mean'][-29:].reshape(1, -1)
mape = mean_absolute_percentage_error(test, pred)
plt.plot(X_train['Date'], X_train['Counts'], label='Train')
plt.plot(X_test['Date'], X_test['Counts'], label='Test')
plt.plot(X['Date'][-29:], X['Simple_mean'][-29:], label='Simple_mean')
plt.title('TPCC8131_mape:{}'.format(mape))
plt.legend(loc='upper left')
my_xticks = df.Date
frequency = 20
plt.xticks(df.Date[::frequency], my_xticks[::frequency], rotation=45)
plt.grid()
plt.show()


