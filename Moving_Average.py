from util import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import check_array


plt.plot(x_train.mean(axis=1), label='pred')
plt.plot(y_train, 'r-', label='real')
plt.title('TPCC8131')
plt.legend()
plt.grid()
plt.show()

# --------------------------------------------------- plot ---------------------------------------------------------
mape_train = mean_absolute_percentage_error(y_train.values.reshape(1, -1), x_train.mean(axis=1).reshape(1,-1))
plt.plot(x_train.mean(axis=1), 'r-', label='predict')
plt.plot(y_train, label='real')
plt.title('TPCC8131_MA_train_mape: {}'.format(mape_train))
plt.legend(loc='upper left')
plt.grid()
plt.show()

mape_valid = mean_absolute_percentage_error(y_valid.values.reshape(1, -1), x_valid.mean(axis=1).reshape(1,-1))
plt.plot(x_valid.mean(axis=1), 'r-', label='predict')
plt.plot(y_valid, label='real')
plt.title('TPCC8131_MA_valid_mape: {}'.format(mape_valid))
plt.legend(loc='upper left')
plt.grid()
plt.show()

mape_test = mean_absolute_percentage_error(y_test.values.reshape(1, -1), x_test.mean(axis=1).reshape(1,-1))
plt.plot(x_test.mean(axis=1), 'r-', label='predict')
plt.plot(y_test, label='real')
plt.title('TPCC8131_MA_test_mape: {}'.format(mape_test))
plt.legend(loc='upper left')
plt.grid()
plt.show()
