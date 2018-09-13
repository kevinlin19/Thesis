import pandas as pd
from sklearn.utils import check_array
import numpy as np

# names = ['x_0', 'x_1','x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'y_15']
df = pd.read_csv('./result_eight_0706.csv', header=0, index_col=0)
# training 50, valid 50, test 50
x_train = df.iloc[:55, :8]
y_train = df.iloc[:55, 8]
x_valid = df.iloc[55:110, :8]
y_valid = df.iloc[55:110, 8]
x_test = df.iloc[110:, :8]
y_test = df.iloc[110:, 8]
# Scaling
X_scale_max = df.iloc[:110, :8].max().max()
X_scale_min = df.iloc[:110, :8].min().min()
diff_X = X_scale_max - X_scale_min
def X_scale(data):
    scale = 100*(data - X_scale_min)/diff_X
    return scale
Y_scale_max = df.iloc[:100, 8].max().max()
Y_scale_min = df.iloc[:100, 8].min().min()
diff_Y = Y_scale_max - Y_scale_min
def Y_scale(data):
    scale_Y = 100*(data - Y_scale_min)/diff_Y
    return scale_Y

# data scale fit
x_train_scale = X_scale(x_train)
y_train_scale = Y_scale(y_train)
x_valid_scale = X_scale(x_valid)
y_valid_scale = Y_scale(y_valid)
x_test_scale = X_scale(x_test)
y_test_scale = Y_scale(y_test)


def X_scale_inverse(data):
    X_inverse = (diff_X * data / 100) + X_scale_min
    return X_inverse
def Y_scale_inverse(data):
    Y_inverse = (diff_Y * data / 100) + Y_scale_min
    return Y_inverse

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

def MASE(y_true, y_pred):
    y_true = check_array(y_true.reshape(-1, 1))
    y_pred = check_array(y_pred.reshape(-1, 1))
    mase = np.mean(np.abs(y_true - y_pred)/np.mean(np.abs(y_true[1:] - y_pred[:-1])))
    return mase

def AIC(model, y_true, y_pred):
    number_params = model.count_params()
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    resid = y_true - y_pred
    sse = np.sum(resid**2)
    aic = 2*number_params - 2*np.log(sse)
    return aic

def BIC(model, y_true, y_pred):
    number_params = model.count_params()
    observation = len(y_true)
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    resid = y_true - y_pred
    sse = np.sum(resid**2)
    bic = number_params*np.log(sse/number_params) + observation*np.log(number_params)
    return bic

def SMAPE(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    smape = np.mean(np.abs(y_true - y_pred) * 2/(y_true + y_pred))
    return smape

def MADP(y_true, y_pred):
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    madp = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    return madp


# test git

