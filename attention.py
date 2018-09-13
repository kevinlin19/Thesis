from util import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, LeakyReLU, BatchNormalization, merge
from keras.models import Sequential, Input, Model
from keras.layers.core import *
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from keras.models import load_model


SHIFT = 8
BATCH = 128

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
m.shape # (30, 8, 1)
n.shape # (30, 1)

def attention_3d_block(inputs, num):
    a = Permute((2, 1))(inputs) # (batch, feature, time_step)
    a = Dense(8, activation='softmax')(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction_{}'.format(num))(a)
    a = RepeatVector(int(inputs.shape[2]))(a)
    a_probs = Permute((2, 1), name='attention_vec_{}'.format(num))(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul_{}'.format(num), mode='mul')
    return output_attention_mul
def model_attention_applied_after_lstm():
    inputs = Input(shape=(8, 1))
    lstm_out = LSTM(30, return_sequences=True, activation='relu')(inputs)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1)(attention_mul)
    output = BatchNormalization()(output)
    output = LeakyReLU()(output)
    model = Model(input=[inputs], output=output)
    return model
def model_attention_applied_before_lstm():
    inputs = Input(shape=(8, 1))
    attention_mul = attention_3d_block(inputs, 1)
    attention_mul = LSTM(30, return_sequences=False, activation='relu')(attention_mul)
    output = Dense(1, activation='relu')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

def model_attention_applied_before_after_lstm():
    inputs = Input(shape=(8, 1))
    attention_mul = attention_3d_block(inputs, 1)
    attention_mul = LSTM(1, return_sequences=True, activation='relu')(attention_mul)
    # lstm_out = LSTM(64, return_sequences=True, activation='relu')(attention_mul)
    # attention_mul = attention_3d_block(lstm_out, 2)
    attention_mul = attention_3d_block(attention_mul, 2)
    output = LSTM(30, return_sequences=False, activation='relu')(attention_mul)
    # attention_mul = Flatten()(output)
    output = Dense(1)(output)
    # output = BatchNormalization()(output)
    output = LeakyReLU()(output)
    output = Dense(1, activation='relu')(output)
    model = Model(input=[inputs], output=output)
    return model

model_attention = model_attention_applied_before_lstm()
model_attention = model_attention_applied_before_after_lstm()
model_attention.summary()
model_attention.compile(optimizer='RMSProp', loss='mse', metrics=['mape'])
callback = TensorBoard(batch_size=BATCH)
history = model_attention.fit_generator(generator=generator, steps_per_epoch=30, epochs=1000, callbacks=[callback], validation_data=[x_valid_scale.values.reshape(55 ,8, 1), y_valid_scale.values.reshape(55, 1)])

model_attention = load_model('./model_attention.h5')
pred = model_attention.predict(x_valid_scale.values.reshape(50, 8, 1))
pred_inverse = Y_scale_inverse(pred)
mean_absolute_percentage_error(y_valid.values.reshape(-1, 1), pred_inverse.reshape(-1, 1))

plt.plot(y_valid.values, 'r-', label='real')
plt.plot(pred_inverse, label='pred')
plt.legend()
plt.title('TPCC8131')
plt.show()
model_attention.save('./model_attention.h5')


# --------------------------------------------------- plot ---------------------------------------------------------
attention_model = load_model('./model_attention.h5')
pred_train = attention_model.predict(x_train_scale.values.reshape(50, 8, 1))
pred_train = Y_scale_inverse(pred_train)
mape_train = mean_absolute_percentage_error(y_train.values.reshape(1, -1), pred_train)
plt.plot(pred_train, 'r-', label='predict')
plt.plot(y_train, label='real')
plt.title('TPCC8131_attention_train_mape: {}'.format(mape_train))
plt.legend(loc='upper left')
plt.grid()
plt.show()

pred_valid = model_attention.predict(x_valid_scale.values.reshape(55, 8, 1))
pred_valid = Y_scale_inverse(pred_valid)
mape_valid = mean_absolute_percentage_error(y_valid.values.reshape(1, -1), pred_valid)
plt.plot(pred_valid, 'r-', label='predict')
plt.plot(y_valid.values, label='real')
plt.title('TPCC8131_attention_valid_mape: {}'.format(mape_valid))
plt.legend(loc='upper left')
plt.grid()
plt.show()

pred_test = attention_model.predict(x_test_scale.values.reshape(29, 8, 1))
pred_test = Y_scale_inverse(pred_test)
mape_test = mean_absolute_percentage_error(y_test.values.reshape(1, -1), pred_test)
plt.plot(pred_test, 'r-', label='predict')
plt.plot(y_test.values, label='real')
plt.title('TPCC8131_attention_test_mape: {}'.format(mape_test))
plt.legend(loc='upper left')
plt.grid()
plt.show()