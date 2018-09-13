from util import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt


holt = [Holt(x_train.values[i, :]).fit(smoothing_level=0.3, smoothing_slope=0.1).forecast(8)[-1] for i in range(len(x_train))]

plt.plot(y_train, 'r-', label='real')
plt.plot(holt, label='pred')
plt.title('TPCC8131')
plt.legend()
plt.grid()
plt.show()

# --------------------------------------------------- plot ---------------------------------------------------------
holt_train = [Holt(x_train.values[i, :]).fit(smoothing_level=0.3, smoothing_slope=0.1).forecast(8)[-1] for i in range(len(x_train))]
mape_train = mean_absolute_percentage_error(y_train.values.reshape(1, -1), np.array(holt_train).reshape(1, -1))
plt.plot(holt_train, 'r-', label='predict')
plt.plot(y_train, label='real')
plt.title('TPCC8131_Holt_train_mape: {}'.format(mape_train))
plt.legend(loc='upper left')
plt.grid()
plt.show()

holt_valid = [Holt(x_valid.values[i, :]).fit(smoothing_level=0.3, smoothing_slope=0.1).forecast(8)[-1] for i in range(len(x_valid))]
mape_valid = mean_absolute_percentage_error(y_valid.values.reshape(1, -1), np.array(holt_valid).reshape(1, -1))
plt.plot(holt_valid, 'r-', label='predict')
plt.plot(y_valid.values, label='real')
plt.title('TPCC8131_Holt_valid_mape: {}'.format(mape_valid))
plt.legend(loc='upper left')
plt.grid()
plt.show()

holt_test = [Holt(x_test.values[i, :]).fit(smoothing_level=0.3, smoothing_slope=0.1).forecast(8)[-1] for i in range(len(x_test))]
mape_test = mean_absolute_percentage_error(y_test.values.reshape(1, -1), np.array(holt_test).reshape(1, -1))
plt.plot(holt_test, 'r-', label='predict')
plt.plot(y_test.values, label='real')
plt.title('TPCC8131_Holt_test_mape: {}'.format(mape_test))
plt.legend(loc='upper left')
plt.grid()
plt.show()
