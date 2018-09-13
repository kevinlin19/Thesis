import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

SHIFT = 8
df = pd.read_csv('./week.csv', names=['Date', 'Counts']) # (144, 2)
X = df.Counts[:-8] # 136
Y = df.Counts[15:] # 129
# X = X.reshape((-1, 1))
# Y = Y.reshape((-1, 1))
X_train = X[:int(X.shape[0]*0.8)]
X_test = X[int(X.shape[0]*0.8):]
# x_train.shape # 108
# x_test.shape # 28
Y_train = Y[:108-7] # 101
Y_test = Y[-(28-7):] # 21
x_train = np.zeros(shape=[101, 1])
x_test = np.zeros(shape=[21, 1])
for i in range(101):
    x_train[i] = X_train[i:i+8].mean()
for i in range(21):
    x_test[i] = X_test[i:i+8].mean()
# X_scale = MinMaxScaler(feature_range=(0, 100))
# x_train_scale = X_scale.fit_transform(x_train)
# Y_scale = MinMaxScaler(feature_range=(0, 100))
# y_train_scale = Y_scale.fit_transform(Y_train)


svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
svr.fit(x_train, Y_train)
y_pred = svr.predict(x_train)
y_pred

import matplotlib.pyplot as plt
plt.plot(y_pred)
plt.plot(Y_train, 'r-')
plt.show()