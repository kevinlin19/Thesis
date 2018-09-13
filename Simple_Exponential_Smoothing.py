from util import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import check_array
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

ses = [SimpleExpSmoothing(x_train.values[i, :]).fit(smoothing_level=0.6, optimized=False).forecast() for i in range(len(x_train))]
plt.plot(ses, label='pred')
plt.plot(y_train.values, label='real')
plt.title('TPCC8131')
plt.legend(loc='upper left')
plt.grid()
plt.show()


# --------------------------------------------------- plot ---------------------------------------------------------
ses_train = [SimpleExpSmoothing(x_train.values[i, :]).fit(smoothing_level=0.6, optimized=True).forecast() for i in range(len(x_train))]
mape_train = mean_absolute_percentage_error(y_train.values.reshape(1, -1), ses_train)
plt.plot(ses_train, 'r-', label='predict')
plt.plot(y_train, label='real')
plt.title('TPCC8131_SES_train_mape: {}'.format(mape_train))
plt.legend(loc='upper left')
plt.grid()
plt.show()

ses_valid = [SimpleExpSmoothing(x_valid.values[i, :]).fit(smoothing_level=0.6, optimized=False).forecast() for i in range(len(x_valid))]
mape_valid = mean_absolute_percentage_error(y_valid.values.reshape(1, -1), ses_valid)
plt.plot(ses_valid, 'r-', label='predict')
plt.plot(y_valid.values, label='real')
plt.title('TPCC8131_SES_valid_mape: {}'.format(mape_valid))
plt.legend(loc='upper left')
plt.grid()
plt.show()

ses_test = [SimpleExpSmoothing(x_test.values[i, :]).fit(smoothing_level=0.6, optimized=False).forecast() for i in range(len(x_test))]
mape_test = mean_absolute_percentage_error(y_test.values.reshape(1, -1), ses_test)
plt.plot(ses_test, 'r-', label='predict')
plt.plot(y_test.values, label='real')
plt.title('TPCC8131_SES_test_mape: {}'.format(mape_test))
plt.legend(loc='upper left')
plt.grid()
plt.show()

