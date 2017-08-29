import os
import numpy as np
import pandas as pd
import pickle
import quandl
import json
import requests
from datetime import datetime
import ssl
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates

style_list = ['seaborn-white', 'ggplot', 'seaborn-notebook', 'fivethirtyeight', 'seaborn-colorblind', '_classic_test', 'seaborn-bright', 'seaborn-whitegrid', 'seaborn-darkgrid', 'seaborn-paper', 'dark_background', 'seaborn-poster', 'seaborn-talk', 'seaborn', 'seaborn-deep', 'seaborn-muted', 'bmh', 'seaborn-ticks', 'grayscale', 'seaborn-dark-palette', 'seaborn-pastel', 'seaborn-dark', 'classic']
plt.style.use(style_list[1])

ssl._create_default_https_context = ssl._create_unverified_context



'''def get_quandl_data(quandl_id):
    #Download and cache Quandl dataseries
    cache_path = '{}.pkl'.format(quandl_id).replace('/','-')
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(quandl_id))
    except (OSError, IOError) as e:
        print('Downloading {} from Quandl'.format(quandl_id))
        df = quandl.get(quandl_id, returns="pandas")
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(quandl_id, cache_path))
    return df'''

def get_quandl_data(quandl_id):
    # Download directy from the API with no caching - API limits apply
    print('Downloading {} from Quandl'.format(quandl_id))
    df = quandl.get(quandl_id, returns="pandas")
    return df

# Pull Kraken BTC price exchange data
#btc_usd_price_kraken = get_quandl_data('BCHARTS/KRAKENUSD')

# Pulling BTC from other exchanges



exchange_data = {}
exchanges = ['COINBASE','BITSTAMP','ITBIT','KRAKEN']

#exchange_data['KRAKEN'] = btc_usd_price_kraken

for exchange in exchanges:
    exchange_code = 'BCHARTS/{}USD'.format(exchange)
    btc_exchange_df = get_quandl_data(exchange_code)
    exchange_data[exchange] = btc_exchange_df

def merge_dfs_on_column(dataframes, labels, col):
    #Merge a single column of each dataframe into a new combined dataframe
    series_dict = {}
    for index in range(len(dataframes)):
        series_dict[labels[index]] = dataframes[index][col]

    return pd.DataFrame(series_dict)

# Merge the BTC price dataseries' into a single dataframe
btc_usd_datasets = merge_dfs_on_column(list(exchange_data.values()), list(exchange_data.keys()), 'Weighted Price')


#The next logical step is to visualize how these pricing datasets compare. For this, we'll define a helper function to provide a single-line command to generate a graph from the dataframe.
'''def df_scatter(df, title, seperate_y_axis=False, y_axis_label='', scale='linear', initial_hide=False):
    # Generate a scatter plot of the entire dataframe
    label_arr = list(df)
    series_arr = list(map(lambda col: df[col], label_arr))

    fig, ax = plt.subplots()
    ax.set_title(title, size=10)

    x=np.array(df.tail(30).index)
    y=np.array(df[exchanges[0]].tail(30))
    y2=np.array(df[exchanges[1]].tail(30))
    y3=np.array(df[exchanges[2]].tail(30))
    y4=np.array(df[exchanges[3]].tail(30))

    plot1, = plt.plot(x,y,linewidth=2, color='red')
    plot2, = plt.plot(x,y2, linewidth=2, color='blue')
    plot3, = plt.plot(x,y3,linewidth=2, color='green')
    plot4, = plt.plot(x,y4, linewidth=2, color='black')

    ax.legend((plot1,plot2,plot3,plot4),(label_arr[0],label_arr[1],label_arr[2],label_arr[3]))

    plt.show()'''

# Remove "0" values
btc_usd_datasets.replace(0, np.nan, inplace=True)

# Calculate the average BTC price as a new column
btc_usd_datasets['avg_btc_price_usd'] = btc_usd_datasets.mean(axis=1)

# the final BTC graph based on the average of the 4 exchanges:
'''myFmt = mdates.DateFormatter('%d.%m.')
fig, ax = plt.subplots()
plt.title("BTC mean price", size=10)
x=np.array(btc_usd_datasets.tail(100).index)
y=np.array(btc_usd_datasets['avg_btc_price_usd'].tail(100))
ax.text(0.9,0.5,('Current BTC\nprice: %s USD' % int(btc_usd_datasets['avg_btc_price_usd'].iloc[-1])),
        horizontalalignment='center',
        transform=ax.transAxes,
        fontsize=8)
ax.xaxis.set_major_formatter(myFmt)
plot = plt.plot(x,y,linewidth=2, color='red')'''

#plt.show()


#getting Altcoin data

'''def get_json_data(json_url, cache_path):
    #Download and cache JSON data, return as a dataframe.
    try:
        f = open(cache_path, 'rb')
        df = pickle.load(f)
        print('Loaded {} from cache'.format(json_url))
    except (OSError, IOError) as e:
        print('Downloading {}'.format(json_url))
        df = pd.read_json(json_url)
        df.to_pickle(cache_path)
        print('Cached {} at {}'.format(json_url, cache_path))
    return df'''

def get_json_data(json_url, cache_path):
    # Download directy from the API with no caching - API limits apply
    print('Downloading {}'.format(json_url))
    df = pd.read_json(json_url)
    return df

base_polo_url = 'https://poloniex.com/public?command=returnChartData&currencyPair={}&start={}&end={}&period={}'
start_date = datetime.strptime('2015-01-01', '%Y-%m-%d') # get data from the start of 2015
end_date = datetime.now() # up until today
period = 86400 # pull daily data (86,400 seconds per day)

def get_crypto_data(poloniex_pair):
    # Retrieve cryptocurrency data from poloniex
    json_url = base_polo_url.format(poloniex_pair, start_date.timestamp(), end_date.timestamp(), period)
    data_df = get_json_data(json_url, poloniex_pair)
    data_df = data_df.set_index('date')
    return data_df

altcoins = ['ETH','LTC','XRP','ETC','STR','DASH','SC','XMR','XEM']

altcoin_data = {}
for altcoin in altcoins:
    coinpair = 'BTC_{}'.format(altcoin)
    crypto_price_df = get_crypto_data(coinpair)
    altcoin_data[altcoin] = crypto_price_df


# Calculate USD Price as a new column in each altcoin dataframe
for altcoin in altcoin_data.keys():
    altcoin_data[altcoin]['price_usd'] =  altcoin_data[altcoin]['weightedAverage'] * btc_usd_datasets['avg_btc_price_usd']

# Merge USD price of each altcoin into single dataframe
combined_df = merge_dfs_on_column(list(altcoin_data.values()), list(altcoin_data.keys()), 'price_usd')

# Add BTC price to the dataframe
combined_df['BTC'] = btc_usd_datasets['avg_btc_price_usd']


# Chart everything:
def chart_final_graph(log, btc, period):
    myFmt = mdates.DateFormatter('%d.%m.')
    fig, ax = plt.subplots()
    ax.set_title('Crypto prices', size=10)
    x=np.array(combined_df.tail(period).index)
    color = ['blue','red','green','yellow','#FFA54F', '#6B6B6B', '#00FF7F', '#20B2AA', '#8A360F',
    '#836FFF', '#EEC591', '#708090', '#4A4A4A', '#CD6090', '#8470FF', '#00EEEE', '#7A67EE', '#CD00CD']
    for index,column in enumerate(combined_df):
        if btc == "yes":
            y = combined_df[column].tail(period)
            plot = ax.plot(x,y,linewidth=2, color=color[index], label=column)
        else:
            if column != "BTC":
                y = combined_df[column].tail(100)
                plot = ax.plot(x,y,linewidth=2, color=color[index], label=column)
    ax.xaxis.set_major_formatter(myFmt)
    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',
        left='off',
        #labelbottom='off'  # ticks along the bottom edge are off
        top='off',         # ticks along the top edoy'f
        )
    if log == "yes":
        plt.yscale('log') #logarithmic scale
    ax.legend(loc='upper left')
    plt.show()



chart_final_graph(log="yes", btc="yes", period=30)
