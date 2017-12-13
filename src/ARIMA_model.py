# https://datascience.ibm.com/exchange/public/entry/view/815137c868b916821dec777bdc23013c
import pandas as pd
import numpy as np
import statsmodels.api as sm
import requests
from io import StringIO
import time, json
from datetime import date
import statsmodels
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

def data_clean(df):
    df.fillna(0, inplace=True)
    # indexed_df = df.set_index('date')
    indexed_df = df.sort_values(['date', 'zip_code', 'lookup']).set_index('date')
    ts = indexed_df['temperature']
    # print('{}: {}'.format(zips, ts.shape))
    ts = ts[np.abs(ts-ts.mean())<=(3*ts.std())]
    ts = ts.resample('W').mean()
    ts.dropna(inplace=True)
    ts_log = np.log(ts)
    ts_log = ts_log.replace([np.inf, -np.inf], np.nan)
    ts_log.dropna(inplace=True)
    return ts, ts_log

def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=52,center=False).mean()
    rolstd = timeseries.rolling(window=52,center=False).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue',label='Original')
    mean = plt.plot(rolmean.index.to_pydatetime(), rolmean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolstd.index.to_pydatetime(), rolstd.values, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


def acf_pacf(timeseries):
    #ACF and PACF plots

    lag_acf = acf(timeseries, nlags=10)
    lag_pacf = pacf(timeseries, nlags=10, method='ols')

    #Plot ACF:
    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()

def arm(timeseries, p, q):
    model = ARIMA(timeseries, order=(p, 1, q))
    results_ARIMA = model.fit(disp=-1)
    ts_log_diff = timeseries - timeseries.shift()
    # model = ARIMA(ts_week_log, order=(2, 1, 1))
    # results_ARIMA = model.fit(disp=-1)
    plt.plot(ts_log_diff.index.to_pydatetime(), ts_log_diff.values)
    plt.plot(ts_log_diff.index.to_pydatetime()[1::], results_ARIMA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff[1::])**2))
    plt.show()
    print(results_ARIMA.summary())
    # plot residual errors
    residuals = pd.DataFrame(results_ARIMA.resid)
    residuals.plot(kind='kde')
    print(residuals.describe())
    plt.show
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    print (predictions_ARIMA_diff.head())

def finalize(timeseries):
    size = int(len(timeseries) - 15)
    train, test = timeseries[0:size], timeseries[size:len(timeseries)]
    history = [x for x in train]
    predictions = list()

    print('Printing Predicted vs Expected Values...')
    print('\n')
    for t in range(len(test)):
        model = ARIMA(history, order=(2,1,1))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(float(yhat))
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (np.exp(yhat), np.exp(obs)))

    error = mean_squared_error(test, predictions)

    print('\n')
    print('Printing Mean Squared Error of Predictions...')
    print('Test MSE: %.6f' % error)

    predictions_series = pd.Series(predictions, index = test.index)


if __name__ == '__main__':
    df = pd.read_pickle('df_original.pkl')
    ts_day, ts_day_log = data_clean(df)
    test_stationarity(ts_day_log)
    acf_pacf(ts_day_log)
    arm(ts_day_log, 3, 2)
    finalize(ts_day_log)

#
#
# ts_log_diff = ts_log - ts_log.shift()
# plt.plot(ts_log_diff.index.to_pydatetime(), ts_log_diff.values)
