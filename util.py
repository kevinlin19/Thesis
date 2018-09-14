import pandas as pd
from sklearn.utils import check_array
import numpy as np


def X_scale(data,  X_scale_min, diff_X):
    scale = 100*(data - X_scale_min)/diff_X
    return scale

def Y_scale(data, Y_scale_min, diff_Y):
    scale_Y = 100*(data - Y_scale_min)/diff_Y
    return scale_Y

def read_file(file):
    # names = ['x_0', 'x_1','x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'y_15']
    df = pd.read_csv(file, header=0, index_col=0)
    # training 50, valid 50, test 50
    x_train = df.iloc[:52, :8]
    y_train = df.iloc[:52, 8]
    x_valid = df.iloc[52:92, :8]
    y_valid = df.iloc[52:92, 8]
    x_test = df.iloc[92:, :8]
    y_test = df.iloc[92:, 8]
    x_RL = df.iloc[52:, :8]
    y_RL = df.iloc[52:, 8]
    # Scaling
    X_scale_max = df.iloc[:, :8].max().max()
    X_scale_min = df.iloc[:, :8].min().min()
    diff_X = X_scale_max - X_scale_min
    Y_scale_max = df.iloc[:, 8].max().max()
    Y_scale_min = df.iloc[:, 8].min().min()
    diff_Y = Y_scale_max - Y_scale_min
    # data scale fit
    x_train_scale = X_scale(x_train, X_scale_min, diff_X)
    y_train_scale = Y_scale(y_train, Y_scale_min, diff_Y)
    x_valid_scale = X_scale(x_valid, X_scale_min, diff_X)
    y_valid_scale = Y_scale(y_valid, Y_scale_min, diff_Y)
    x_test_scale = X_scale(x_test, X_scale_min, diff_X)
    y_test_scale = Y_scale(y_test, Y_scale_min, diff_Y)
    x_RL_scale = X_scale(x_RL, X_scale_min, diff_X)
    y_RL_scale = Y_scale(y_RL, Y_scale_min, diff_Y)
    return diff_X, diff_Y, X_scale_min, Y_scale_min, x_train_scale, y_train_scale, x_valid_scale, y_valid_scale, x_test_scale, y_test_scale, x_RL_scale, y_RL_scale

def X_scale_inverse(data, diff_X, X_scale_min):
    X_inverse = (diff_X * data / 100) + X_scale_min
    return X_inverse
def Y_scale_inverse(data, diff_Y, Y_scale_min):
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


def reject_outliers(data, m = 6., outlierConstant=1.5):
    # d = np.abs(data - np.median(data))
    # mdev = np.median(d)
    # s = d/mdev if mdev else 0.
    roll = data.rolling(window=3).mean()
    upper_quartile = np.percentile(data, 75)
    lower_quartile = np.percentile(data, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    # print(IQR)
    # print(quartileSet[0])
    # print(quartileSet[1])
    # data[s < m] = data
    outlier_index = data[data > quartileSet[1]].index
    # print(outlier_index)
    data[outlier_index] = roll[outlier_index]
    # print(data[outlier_index])
    return data
