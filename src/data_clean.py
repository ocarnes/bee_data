import pandas as pd
from requests import get
import pandas as pd
import numpy as np
import json
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import os
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm

def hive_data(filename):
    df = pd.read_csv(filename,
        parse_dates=[['date', 'time', 'ampm']],
        converters={'zip_code': lambda x: '0'+str(x) if len(str(x)) == 4 else str(x)})

    df = df.drop_duplicates()
    df = df.sort_values(['date_time_ampm'])#.reset_index(drop=True)
    df = df[np.abs(df.temperature-df.temperature.mean())<=(4*df.temperature.std())]
    df['lookup'].fillna(value=df.zip_code, inplace = True)
    df = df.drop('lookup2', axis = 1)
    df = df.rename(index=str, columns={'date_time_ampm': 'date'})
    return df

def weather_data(zip_code):
    url = 'https://www.wunderground.com/history/zipcode/{}/2016/4/11/CustomHistory.html?dayend=13&monthend=4&yearend=2017&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo='.format(zip_code)
    response = get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tag = soup.find_all('table')[1]

    df = pd.DataFrame([])
    for line in tag.findAll('tbody'):
        lst = ['/'.join(str(line.a).split('/')[4:7])]
        if len(lst[0]) > 0:
            for val in line.find_all('span', "wx-value"):
                lst.append(val.string)
            df = df.append([lst])
    df = df.rename(index=str, columns={0: "date", 1: "t_max", 2: "t_avg",
    3: "t_min", 7: "h_max", 8: "h_avg", 9: "h_min"})
    df.date = pd.to_datetime(df['date'])
    return df

def df_merge(df_hive):
    count = 0
    zips = list(df_hive.zip_code.unique())
    df = pd.DataFrame([])
    for code in zips:
        df_weather = weather_data(code)
        df_code = df_hive[df_hive['zip_code'] == code]
        df_rev = pd.merge_asof(df_code, df_weather, on='date', direction='nearest')
        df = df.append(df_rev)
        count += 1
        print('{}/{}'.format(count, len(zips)))
    return df

def summarize(df):
    description = df.groupby(['date', 'lookup'])['temperature'].describe().reset_index()
    df = pd.merge(description, df, on=['date','lookup'])
    df.drop('temperature', axis=1, inplace=True)
    df = df.drop_duplicates()
    return df

def regression(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    print('Coefficients: \n', regr.coef_)
    print('Mean squared error: {}'.format(mean_squared_error(y_test, y_pred)))
    print('Variance score: {}'.format(r2_score(y_test, y_pred)))
    model = sm.OLS(y, X).fit()
    summary = model.summary()
    print(summary)

def plotz(df, column, max_date):
    for zips in df[column].unique():
        plt.plot_date(df[(df.date >= max_date)&(df[column] == zips)].date,
         df[(df.date >= max_date)&(df[column]==zips)]['temperature'],
         marker='.', alpha = 0.25)
    # plt.legend()
    plt.axhspan(ymin=89.6, ymax=95, xmin=0, xmax=1, label='Ideal Temp')
    plt.xticks(rotation=45)
    plt.title('Temperature by Date and Zip Code')
    plt.xlabel('Date')
    plt.ylabel('Temp. (F)')
    plt.tight_layout()
    plt.savefig('../img/Hive_Data_by_Zip')
    plt.show()

if __name__ == '__main__':
    filename = '../data/one_hive_data.csv'
    if os.path.exists('../data/df_hive.pkl'):
        df_hive = pd.read_pickle('../data/df_hive.pkl')
    else:
        df_hive = hive_data(filename)
    if os.path.exists('../data/df_weather.pkl'):
        df_weather = pd.read_pickle('../data/df_weather.pkl')
    else:
        df_combined = df_merge(df_hive)
        df_weather = df_combined.dropna()
        df_weather.iloc[:, 6:23] = df_weather.iloc[:,6:23].apply(pd.to_numeric)
        df_weather.drop([ 18, 19], axis=1, inplace=True)
    df = pd.get_dummies(df_weather, columns=['lookup'], prefix='zip_')
    # y = df_weather.pop('temperature').values
    # X = pd.get_dummies(df_weather, columns=['lookup'], prefix='zip_').iloc[:,5::].values
    # regression(X,y)
    df_sum = summarize(df_weather)
    plotz(df_hive, 'zip_code', '2016-04-01')
