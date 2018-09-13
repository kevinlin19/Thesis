from util import *
from DQN import DeepQNetwork
from keras.models import load_model
import pickle
import xgboost as xgb
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_array




# method = {0:'lstm', 1:'attention', 2:'xgb', 3:'Naive', 4:'MA', 5:'ses', 6:'HL'}

lstm_model = load_model('./lstm_model.h5')
attention_model = load_model('./model_attention.h5')
xgb_model = pickle.load(open("pima.pickle.dat", "rb"))

def run_this():
    step = 0
    methods = []
    prediction = []
    for date in range(50):
        # observation = x_valid.values[date]
        # observation_scale = x_valid_scale[date]
        action = RL.choose_action(x_valid.values[date])
        methods.append(action)
        if action == 0:
            pred = lstm_model.predict(x_valid_scale.values[date].reshape(1, 8, 1))
            pred = Y_scale_inverse(pred)
            reward = -1 * np.abs(y_valid.values[date] - pred)
        elif action == 1:
            pred = attention_model.predict(x_valid_scale.values[date].reshape(1, 8, 1))
            pred = Y_scale_inverse(pred)
            reward = -1 * np.abs(y_valid.values[date] - pred)
        elif action == 2:
            pred = x_valid.values[date][-1]
            reward = -1 * np.abs(y_valid.values[date] - pred)

        elif action == 3:
            pred = x_valid.values[date].mean()
            reward = -1 * np.abs(y_valid.values[date] - pred)

        elif action == 4:
            pred = SimpleExpSmoothing(x_valid.values[date]).fit(smoothing_level=0.6, optimized=False).forecast()
            reward = -1 * np.abs(y_valid.values[date] - pred)

        elif action == 5:
            pred = Holt(x_valid.values[date]).fit(smoothing_level=0.3, smoothing_slope=0.1).forecast(8)[-1]
            reward = -1 * np.abs(y_valid.values[date] - pred)

        elif action == 6:
            pred = xgb_model.predict(xgb.DMatrix(x_valid.values[date].reshape(1, -1)))
            reward = -1 * np.abs(y_valid.values[date] - pred)

        prediction.append(pred)

        if np.abs(y_valid.values[date] - pred) > 20000:
            reward = reward - 100000

        elif np.abs(y_valid.values[date] - pred) < 10000:
            reward = reward + 100000

        try:
            RL.store_transition(x_valid.values[date], action, reward, x_valid.values[date+1])
        except:
            pass

        if (date > 5) and (date % 5 == 0):
            RL.learn()

    print(methods)
    print(prediction)
    return prediction

# if __name__ == '__main__':
RL = DeepQNetwork(7, 8)
for i in range(500):
    p = run_this()

mape = mean_absolute_percentage_error(y_valid.reshape(1, -1), np.array(p).reshape(1, -1))
plt.plot(p, 'r-', label='predict')
plt.plot(y_valid.values, label='real')
plt.title('TPCC8131_RL_valid_mape: {}'.format(mape))
plt.legend(loc='upper left')
plt.grid()
plt.show()
#
