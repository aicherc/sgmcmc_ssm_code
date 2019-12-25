import pandas as pd
import numpy as np
import os

###############################################################################
# Process Raw Exchange Rate Data Pulled From Finam.RU
###############################################################################
path_to_raw_data = './data/EURUS_data.csv'
path_to_output_data = './data/EURUS_processed.npz'


df = pd.read_csv(path_to_raw_data, dtype={'<DATE>': str, '<TIME>':str})
df['date'] = pd.to_datetime(df['<DATE>'] + df['<TIME>'], format="%Y%m%d%H%M%S")
prices = df[['date', '<CLOSE>']]
prices.columns = ['date', 'close']

# Minute Returns
minute_series = pd.DataFrame(prices['close'].map(np.log).diff() - np.mean(prices['close'].map(np.log).diff()))
minute_series['date'] = prices['date']
minute_series = minute_series.iloc[1:]
minute_series.columns = ['log_returns', 'date']

# Hourly Returns
hourly_price = prices.copy()
hourly_price['date'] = hourly_price['date'].map(lambda x: x.replace(minute=0))
hourly_price = hourly_price.groupby(['date']).nth(0).reset_index()
hourly_series = pd.DataFrame(hourly_price['close'].map(np.log).diff() - np.mean(hourly_price['close'].map(np.log).diff()))
hourly_series['date'] = hourly_price['date']
hourly_series = hourly_series.iloc[1:]
hourly_series.columns = ['log_returns', 'date']

# Daily Returns
daily_price = prices.copy()
daily_price['date'] = daily_price['date'].map(lambda x: x.replace(hour=0, minute=0))
daily_price = daily_price.groupby(['date']).nth(0).reset_index()
daily_series = pd.DataFrame(daily_price['close'].map(np.log).diff() - np.mean(daily_price['close'].map(np.log).diff()))
daily_series['date'] = daily_price['date']
daily_series = daily_series.iloc[1:]
daily_series.columns = ['log_returns', 'date']

# Save Data
data = dict(
        minute_log_returns=minute_series['log_returns'],
        minute_date=np.array(minute_series['date'], dtype='datetime64[m]'),
        hourly_log_returns=hourly_series['log_returns'],
        hourly_date=np.array(hourly_series['date'], dtype='datetime64[h]'),
        daily_log_returns=daily_series['log_returns'],
        daily_date=np.array(daily_series['date'], dtype='datetime64[D]'),
        )
np.savez_compressed(path_to_output_data, **data)




