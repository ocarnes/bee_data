# README
## Intro ##
This project was completed to provide recommendations for analyzing temperature data at specific hive locations throughout the country.

### Contents ###
1. Data cleaning
2. Modeling
3. Analysis


## Data Cleaning ##
Data cleaning involved importing a csv file in addition to scraping some info from [NOAA](https://www.ncdc.noaa.gov/cdo-web/webservices/v2#datasets) and [Weather Underground](https://www.wunderground.com/history/) for additional climate data based on the zip codes provided. (Note: The [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) python package was used to parse text scrapped
from these sites).
It was assumed that hive data was recorded either a single location within a zip code or with multiple locations spread across a single zip code. The graph below reflects all of the different hives recording data throughout the year long test period.
![Hive Data by Zip](/img/Hive_Data_by_Zip.png).
Once Nan columns were eliminated and the appropriate additional weather data was merged, preliminary modeling could begin.

## Modeling ##
Autoregressive integrated moving average ([ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)) models and Long Short Term Memory ([LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)) networks were used to perform time-series analysis on the data.

![ARIMA](/img/ARIMA_RSS.png)

The ARIMA model had a test RMSE of 0.1138. The Multivariate LSTM network, on the other hand, had a test RMSE of 10.752.

![LSTM Multivariate](/img/LSTM.png)

Parameters have yet to be played around with though so these numbers will likely improve in the future.

![LSTM Univariate](/img/LSTM_univariate.png)

## Analysis ##
The univariate ARIMA model is currently performing best of the three models based on RMSE values. Future directions would likely include:
1. Regression analysis to fill in missing time series data
2. Categorical variables on hive health
3. Pulling minute by minute rather than day by day outside information
4. Fine-tuning model parameters

### References ###
[ARIMA models]( https://datascience.ibm.com/exchange/public/entry/view/815137c868b916821dec777bdc23013c
)
[RNNs](https://github.com/GalvanizeOpenSource/Recurrent_Neural_Net_Meetup/blob/master/RNN_Meetup_Presentation.pdf)
[Multivariate LSTMs](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
