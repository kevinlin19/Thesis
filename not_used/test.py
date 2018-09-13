# from util import *
# from DQN import DeepQNetwork
# from keras.models import load_model
# import pickle
# import xgboost as xgb
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# from statsmodels.tsa.holtwinters import Holt
# import numpy as np



# method = {0:'lstm', 1:'attention', 2:'xgb', 3:'Naive', 4:'MA', 5:'ses', 6:'HL'}

# lstm_model = load_model('./lstm_model.h5')
# attention_model = load_model('./model_attention.h5')
# xgb_model = pickle.load(open("pima.pickle.dat", "rb"))

def run_this_test():
    methods_test = []
    prediction_test = []
    for date in range(29):
        action = RL.choose_action(x_test.values[date])
        methods_test.append(action)
        if action == 0:
            pred = lstm_model.predict(x_test_scale.values[date].reshape(1, 8, 1))
            pred = Y_scale_inverse(pred)
        elif action == 1:
            pred = attention_model.predict(x_test_scale.values[date].reshape(1, 8, 1))
            pred = Y_scale_inverse(pred)
        elif action == 2:
            pred = x_test.values[date][-1]
        elif action == 3:
            pred = x_test.values[date].mean()
        elif action == 4:
            pred = SimpleExpSmoothing(x_test.values[date]).fit(smoothing_level=0.6, optimized=False).forecast()
        elif action == 5:
            pred = Holt(x_test.values[date]).fit(smoothing_level=0.3, smoothing_slope=0.1).forecast(8)[-1]
        elif action == 6:
            pred = xgb_model.predict(xgb.DMatrix(x_test.values[date].reshape(1, -1)))

        prediction_test.append(pred)

        # try:
        #     RL.store_transition(x_test.values[date], action, reward, x_test.values[date+1])
        # except:
        #     pass

        # if (date > 5) and (date % 5 == 0):
        #     RL.learn()

    print(methods_test)
    print(prediction_test)
    return prediction_test

# if __name__ == '__main__':
# RL = DeepQNetwork(6, 8)
test = run_this_test()

mape = mean_absolute_percentage_error(y_test.reshape(1, -1), np.array(test).reshape(1, -1))
plt.plot(test, 'r-', label='predict')
plt.plot(y_test.values, label='real')
plt.title('TPCC8131_RL_train_mape: {}'.format(mape))
plt.legend(loc='upper left')
plt.grid()
plt.show()