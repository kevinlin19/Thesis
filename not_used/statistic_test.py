from util import *
import statsmodels as sm
import pandas as pd
import statsmodels.tsa.stattools as tsa
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import seaborn as sns

df_test = pd.read_csv('./week.csv', names=['date', 'counts'])
result = tsa.adfuller(df_test.counts, autolag='AIC')
print('Result of Augment Dickary-Fuller test--AIC')
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

'''
Result of Augment Dickary-Fuller test--AIC
ADF Statistic: -3.061452
p-value: 0.029541
Critical Values:
	1%: -3.478
	5%: -2.882
	10%: -2.578

'''

def adf_test(y):
    # perform Augmented Dickey Fuller test
    print('Results of Augmented Dickey-Fuller test:')
    dftest = tsa.adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print(dfoutput)
adf_test(df_test.counts)

# apply the function to the time series
test_for_stationarity(y)
'''
Results of Augmented Dickey-Fuller test:
test statistic           -3.061452
p-value                   0.029541
# of lags                 3.000000
# of observations       140.000000
Critical Value (1%)      -3.477945
Critical Value (5%)      -2.882416
Critical Value (10%)     -2.577902
dtype: float64
'''



def ts_diagnostics(y, lags=None, title='', filename=''):
    '''
    Calculate acf, pacf, qq plot and Augmented Dickey Fuller test for a given time series
    '''
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # weekly moving averages (5 day window because of workdays)
    rolling_mean = pd.rolling_mean(y, window=8)
    rolling_std = pd.rolling_std(y, window=8)

    fig = plt.figure(figsize=(14, 12))
    layout = (3, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    qq_ax = plt.subplot2grid(layout, (2, 0))
    hist_ax = plt.subplot2grid(layout, (2, 1))

    # time series plot
    y.plot(ax=ts_ax)
    rolling_mean.plot(ax=ts_ax, color='crimson');
    rolling_std.plot(ax=ts_ax, color='darkslateblue');
    plt.legend(loc='best')
    ts_ax.set_title(title, fontsize=24);

    # acf and pacf
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)

    # qq plot
    sm.api.qqplot(y, line='s', ax=qq_ax)
    qq_ax.set_title('QQ Plot')

    # hist plot
    y.plot(ax=hist_ax, kind='hist', bins=25);
    hist_ax.set_title('Histogram');
    plt.tight_layout();
    # plt.savefig('./img/{}.png'.format(filename))
    plt.show()

    # perform Augmented Dickey Fuller test
    print('Results of Dickey-Fuller test:')
    dftest = tsa.adfuller(y, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['test statistic', 'p-value', '# of lags', '# of observations'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    return
# 一階差分
y_diff = np.diff(df_test.counts)
ts_diagnostics(y_diff, lags=30, title='TPCC8131 diff', filename='./stattest')
'''Results of Dickey-Fuller test:
test statistic         -7.154822e+00
p-value                 3.074128e-10
# of lags               5.000000e+00
# of observations       1.370000e+02
Critical Value (1%)    -3.479007e+00
Critical Value (5%)    -2.882878e+00
'''
# log
y_log = np.log(df_test.counts[13:])
ts_diagnostics(y_log, lags=30, title='TPCC8131 log', filename='./stattest')

# build scatterplot
ncols = 3
nrows = 3
lags = 9

fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(6 * ncols, 6 * nrows))

for ax, lag in zip(axes.flat, np.arange(1, lags + 1, 1)):
    lag_str = 't-{}'.format(lag)
    X = (pd.concat([df_test.counts, df_test.shift(-lag)], axis=1, keys=['y'] + [lag_str]).dropna())

    # plot data
    X.plot(ax=ax, kind='scatter', y='y', x=lag_str);
    corr = X.corr().as_matrix()[0][1]
    ax.set_ylabel('Original');
    ax.set_title('Lag: {} (corr={:.2f})'.format(lag_str, corr));
    ax.set_aspect('equal');

    # top and right spine from plot
    sns.despine();

fig.tight_layout()
plt.show()

