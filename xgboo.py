from util import *
import numpy as np
import xgboost as xgb
from sklearn.model_selection import  RandomizedSearchCV
from xgboost.sklearn import XGBRegressor # wrapper
import matplotlib.pyplot as plt
import pickle

# evaluation metric: rmspe# evaluat
# Root Mean Square Percentage Error
# code chunk shared at Kaggle

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y-1) ** 2))

def xgb_mape(preds, dtrain):
   labels = dtrain.get_label()
   return('mape', np.mean(np.abs((labels - preds) / (labels+1))))
# base parameters
params = {
    'booster': 'gbtree',
    'objective': 'reg:linear', # regression task
    'subsample': 0.8, # 80% of data to grow trees and prevent overfitting
    'colsample_bytree': 0.85, # 85% of features used
    'eta': 0.1,
    'max_depth': 10,
    'seed': 42} # for reproducible results

# XGB with xgboost library
dtrain = xgb.DMatrix(x_train.values, y_train.values)
dtest = xgb.DMatrix(x_valid.values, y_valid.values)

watchlist = [(dtrain, 'train'), (dtest, 'test')]

xgb_model = xgb.train(params, dtrain, 150*128, evals = watchlist, feval=xgb_mape, verbose_eval = True)

# XGB with sklearn wrapper
# the same parameters as for xgboost model
params_sk = {'max_depth': 10,
            'n_estimators': 300, # the same as num_rounds in xgboost
            'objective': 'reg:linear',
            'subsample': 0.8,
            'colsample_bytree': 0.85,
            'learning_rate': 0.1,
            'seed': 42}
skrg = XGBRegressor(**params_sk)
skrg.fit(x_train.values, y_train.values)

# Grid  search
import scipy.stats as st

params_grid = {
    'learning_rate': st.uniform(0.01, 0.3),
    'max_depth': list(range(10, 20, 2)),
    'gamma': st.uniform(0, 10),
    'reg_alpha': st.expon(0, 50)}

search_sk = RandomizedSearchCV(skrg, params_grid, cv = 5) # 5 fold cross validation
search_sk.fit(x_train.values, y_train.values)
# best parameters
print(search_sk.best_params_); print(search_sk.best_score_)
# {'gamma': 3.768441985810389, 'learning_rate': 0.14786974084341625, 'max_depth': 18, 'reg_alpha': 157.48621463992382}
# -1.0419691781691516
# scale
#{'gamma': 3.8164143858810107, 'learning_rate': 0.09814791191829603, 'max_depth': 16, 'reg_alpha': 9.087071065531077}
# -0.6748372718416444
params_new = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'eta':  0.09814791191829603,
    'max_depth': 16,
    'gamma': 3.768441985810389,
    'reg_alpha': 9.087071065531077,
    'seed': 42}

model_final = xgb.train(params_new, dtrain, num_boost_round=150*128, evals = watchlist, early_stopping_rounds = 50, feval = xgb_mape, verbose_eval = True)
yhat = model_final.predict(xgb.DMatrix(x_valid.values[0].reshape(1, -1)))
# yhat_inverse = Y_scale_inverse(yhat.reshape((-1, 1)))
# plt.plot(yhat_inverse, 'r-')
plt.plot(y_valid.values, 'r-', label='real')
plt.plot(yhat, label='pred')
plt.legend()
plt.show()
xgb.plot_importance(model_final)

pickle.dump(model_final, open("pima.pickle.dat", "wb"))
model_final = pickle.load(open("pima.pickle.dat", "rb"))


xgb_ = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb_.fit(x_train.values, y_train.values)
xgb_.predict(x_valid.values[0].reshape(1, -1))


# --------------------------------------------------- plot ---------------------------------------------------------
xgb_model = pickle.load(open("pima.pickle.dat", "rb"))
pred_train = xgb_model.predict(xgb.DMatrix(x_train.values))
mape_train = mean_absolute_percentage_error(y_train.values.reshape(1, -1), pred_train.reshape(1,-1))
plt.plot(pred_train, 'r-', label='predict')
plt.plot(y_train, label='real')
plt.title('TPCC8131_xgb_train_mape: {}'.format(mape_train))
plt.legend(loc='upper left')
plt.grid()
plt.show()

pred_valid = xgb_model.predict(xgb.DMatrix(x_valid.values))
mape_valid = mean_absolute_percentage_error(y_valid.values.reshape(1, -1), pred_valid.reshape(1, -1))
plt.plot(pred_valid, 'r-', label='predict')
plt.plot(y_valid.values, label='real')
plt.title('TPCC8131_xgb_valid_mape: {}'.format(mape_valid))
plt.legend(loc='upper left')
plt.grid()
plt.show()


pred_test = xgb_model.predict(xgb.DMatrix(x_test.values))
mape_test = mean_absolute_percentage_error(y_test.values.reshape(1, -1), pred_test.reshape(1, -1))
plt.plot(pred_test, 'r-', label='predict')
plt.plot(y_test.values, label='real')
plt.title('TPCC8131_xgb_test_mape: {}'.format(mape_test))
plt.legend(loc='upper left')
plt.grid()
plt.show()