from util import *
import matplotlib.pyplot as plt


plt.plot(y_valid.values, 'r-', label='real')
plt.plot(x_valid.values[:, -1], label='pred')

plt.title('TPCC8131')
plt.legend()
plt.grid()
plt.show()

# --------------------------------------------------- plot ---------------------------------------------------------
mape_train = mean_absolute_percentage_error(y_train.values.reshape(1, -1), x_train.values[:, -1].reshape(1, -1))
plt.plot(x_train.values[:, -1], 'r-', label='predict')
plt.plot(y_train, label='real')
plt.title('TPCC8131_NA_train_mape: {}'.format(mape_train))
plt.legend(loc='upper left')
plt.grid()
plt.show()

mape_valid = mean_absolute_percentage_error(y_valid.values.reshape(1, -1), x_valid.values[:, -1].reshape(1, -1))
plt.plot(x_valid.values[:, -1], 'r-', label='predict')
plt.plot(y_valid.values, label='real')
plt.title('TPCC8131_NA_valid_mape: {}'.format(mape_valid))
plt.legend(loc='upper left')
plt.grid()
plt.show()

mape_test = mean_absolute_percentage_error(y_test.values.reshape(1, -1), x_test.values[:, -1].reshape(1, -1))
plt.plot(x_test.values[:, -1], 'r-', label='predict')
plt.plot(y_test.values, label='real')
plt.title('TPCC8131_NA_test_mape: {}'.format(mape_test))
plt.legend(loc='upper left')
plt.grid()
plt.show()

