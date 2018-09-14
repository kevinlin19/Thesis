'''
116 weeks
52 to train time series
40 to train selection
24 to test
'''
import pandas as pd
import numpy as np
from util import *
import os

# load data
df = pd.read_csv("C:/Users/S.K.LIN/Desktop/project_big_company/data/wCompany_csv.csv")
# To datetime and sort
df['TRANSACTION_DATE'] = pd.to_datetime(df.TRANSACTION_DATE)
df_sort = df.sort_values(by='TRANSACTION_DATE')
# pivot dataframe: row=date, columns=items
df_sort = df_sort.pivot_table(index='TRANSACTION_DATE', columns='ITEM_SHORT_NAME', values='QUANTITY', aggfunc=np.sum)
df_sort['TRANSACTION_DATE'] = df_sort.index
# Fill dates
df_date = pd.DataFrame(pd.date_range(start='2015-04-01', end='2017-12-29'), columns=['TRANSACTION_DATE'])
df_filled = pd.merge(df_date, df_sort, how='left', on='TRANSACTION_DATE')
df_filled = df_filled.fillna(0)
# df_filled['AON7410L']
# df_sort['AON7410L']
first_hundred = (df_filled == 0).astype(int).sum(axis=0).sort_values()[:100].index
'''
The number of 0
AON7410L                             135
TPCC8131                             191
AON7403L                             203
AON7506                              225
AON6520                              247
TPCC8067-H                           271
AOZ8902CIL                           313
AON6510                              323
'''
# pd.date_range(start='2015-04-01', end='2017-12-29')[~pd.date_range(start='2015-04-01', end='2017-12-29').isin(df.sort_values(by='TRANSACTION_DATE')['TRANSACTION_DATE'].unique())].shape
'''
['2015-04-04', '2015-04-05', '2015-04-12', '2015-04-19',
               '2015-04-25', '2015-04-26', '2015-05-01', '2015-05-02',
               '2015-05-03', '2015-05-10', '2015-05-17', '2015-05-24',
               '2015-05-31', '2015-06-07', '2015-06-14', '2015-06-20',
               '2015-06-21', '2015-06-27', '2015-06-28', '2015-08-16',
               '2015-08-23', '2015-09-27', '2015-10-04', '2016-01-17',
               '2016-02-07', '2016-02-08', '2016-04-03', '2016-04-04',
               '2016-06-09', '2016-07-02', '2016-12-31', '2017-01-01',
               '2017-01-08', '2017-01-27', '2017-05-30', '2017-09-03',
               '2017-10-01', '2017-11-05']
'''
# len(df.ITEM_SHORT_NAME.unique()) # 802 item
# len(df.TRANSACTION_DATE.unique()) # 966

# Parameter setting
TARGET = 'TPCC8131'
SHIFT = 14
BATCH = 128
TRAIN_SIZE = 0.8
# df_filled[TARGET].shift(-SHIFT)
for i, TARGET in enumerate(first_hundred):
    df_filled_target = df_filled[['TRANSACTION_DATE', TARGET]]
    try:
        df_test = df_filled_target.groupby([pd.Grouper(key='TRANSACTION_DATE', freq='W-MON')])[TARGET].sum()
        df_test = reject_outliers(df_test)
        # df[df > 800000].index # '2017-01-23', '2017-04-24', '2017-06-12'
        # averaging smoothing
        # df['2017-01-23'] = df[df.index.isin(['2017-01-23', '2017-01-30', '2017-01-16'])].mean()
        # df['2017-04-24'] = df[df.index.isin(['2017-05-01', '2017-04-24', '2017-04-17'])].mean()
        # df['2017-06-12'] = df[df.index.isin(['2017-06-12', '2017-06-19', '2017-06-05'])].mean()
        # save csv
        print(df_test.shape)
        print(i)
        df_test.to_csv('./data_smooth_outliers/week_{}.csv'.format(TARGET))
    except:
        print(i)
        print(TARGET)

path = './data_smooth_outliers'
for file in os.listdir(path):
    file_path = path + '/' + file
    df = pd.read_csv(file_path, names=['Date', 'x_0'])
    df_0706 = df.copy()[13:]
    # df_0706.to_csv('./df_0706.csv')
    # df_0706 = pd.read_csv('./df_0706.csv', names=['Date', 'x_0'])
    # len(df_0706) #131
    # len(df) #144
    x_eight = np.zeros(shape=[len(df_0706)-15, 8]) # (144-15, 8) (131-15, 8)
    for i in range(len(df_0706)-15):
        x_eight[i] = df_0706['x_0'][i: i+8]
    df_eight = pd.DataFrame(x_eight, columns=['x_0', 'x_1','x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7'])
    df_move = pd.DataFrame(df_0706['x_0'].values[15:], columns=['y_15'])
    df_result = pd.concat([df_eight, df_move], axis=1)
    df_result.to_csv('./data_eight_input/result_eight_{}.csv'.format(file))