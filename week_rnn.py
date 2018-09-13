from util import * # import data
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, LSTM, LeakyReLU, BatchNormalization, Activation, Dropout
from keras.models import Sequential, load_model
from keras.callbacks import TensorBoard



# parameter
SHIFT = 8
BATCH = 30
def get_batch(data_x, data_y, batch_size, time_step):
    while True:
        x_batch = np.zeros(shape=[batch_size, time_step, 1], dtype=np.float32)
        y_batch = np.zeros(shape=[batch_size, 1], dtype=np.float32)

        for i in range(batch_size):
            index = np.random.randint(len(data_x))
            x_batch[i] = data_x.values[index].reshape([SHIFT, 1])
            y_batch[i] = data_y.values[index]

        yield (x_batch, y_batch)
generator = get_batch(x_train_scale, y_train_scale, BATCH, SHIFT)
m,n = next(generator)
# m.shape # (30, 8, 1)
# n.shape # (30, 1)

# RNN model 1
# model = Sequential()
# model.add(LSTM(input_shape=(None, 1), units=60, return_sequences=True))
# model.add(LeakyReLU())
# model.add(BatchNormalization())
# model.add(LSTM(input_shape=(None, 1), units=30, return_sequences=False))
# model.add(LeakyReLU())
# model.add(Dense(units=1))
# model.add(Activation('relu'))
# model.summary()
# model 2
model = Sequential()
model.add(LSTM(input_shape=(None, 1), units=60, return_sequences=True))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(LSTM(input_shape=(None, 1), units=30, return_sequences=False))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.add(Activation('relu'))
model.summary()
model.compile(optimizer='RMSProp', loss='mape', metrics=['mse'])
callback = TensorBoard(batch_size=BATCH)
history = model.fit_generator(generator=generator, steps_per_epoch=30, epochs=1000, callbacks=[callback], validation_data=[x_valid_scale.values.reshape(55, 8, 1), y_valid_scale.values.reshape(55 ,1)])
model.save('rnn_model.h5')

pred = model.predict(x_valid_scale.values.reshape(55, 8, 1))
pred_inverse = Y_scale_inverse(pred)
pred_inverse.shape

plt.plot(y_valid.values, 'r-', label='real')
plt.plot(pred_inverse, label='pred')
plt.title('TPCC8131')
plt.legend()
plt.show()

# model.save('./lstm_model.h5')

# --------------------------------------------------- plot ---------------------------------------------------------
lstm_model = load_model('./lstm_model.h5')
pred_train = model.predict(x_train_scale.values.reshape(55, 8, 1))
pred_train = Y_scale_inverse(pred_train)
mape_train = mean_absolute_percentage_error(y_train.values.reshape(1, -1), pred_train)
plt.plot(pred_train, 'r-', label='predict')
plt.plot(y_train, label='real')
plt.title('TPCC8131_train_mape: {}'.format(mape_train))
plt.legend(loc='upper left')
plt.grid()
plt.show()

pred_valid = model.predict(x_valid_scale.values.reshape(55, 8, 1))
pred_valid = lstm_model.predict(x_valid_scale.values.reshape(55, 8, 1))
pred_valid = Y_scale_inverse(pred_valid)
mape_valid = mean_absolute_percentage_error(y_valid.values.reshape(1, -1), pred_valid)
plt.plot(pred_valid, 'r-', label='predict')
plt.plot(y_valid.values, label='real')
plt.title('TPCC8131_valid_mape: {}'.format(mape_valid))
plt.legend(loc='upper left')
plt.grid()
plt.show()

pred_test = lstm_model.predict(x_test_scale.values.reshape(29, 8, 1))
pred_test = Y_scale_inverse(pred_test)
mape_test = mean_absolute_percentage_error(y_test.values.reshape(1, -1), pred_test)
plt.plot(pred_test, 'r-', label='predict')
plt.plot(y_test.values, label='real')
plt.title('TPCC8131_test_mape: {}'.format(mape_test))
plt.legend(loc='upper left')
plt.grid()
plt.show()