
from alphavantage_base import AlphaVantage
import pandas as pd
import json


def get_main_data_frame(symbol):
    #
    # Get historical data as json
    #
    historical_data_daily = AlphaVantage(symbol).daily()

    #
    # Find and list all relevant keys in the original historical data response
    #
    list_keys = []
    list_historical_data__daily = []
    for key in historical_data_daily['Time Series (Daily)']:
        list_keys.append(key)

    for k in list_keys:
        #
        # Extract relevant data from original historical data response
        #
        data = historical_data_daily['Time Series (Daily)'][k]

        price_open = data['1. open']
        price_high = data['2. high']
        price_low = data['3. low']
        price_close = data['4. close']

        dict_data = dict([
            (u'date', k),
            (u'symbol', symbol),
            (u'price_open', price_open),
            (u'price_high', price_high),
            (u'price_low', price_low),
            (u'price_close', price_close),
        ])

        list_historical_data__daily.append(dict_data)

    #
    # convert data to data frame
    #
    df = pd.read_json(json.dumps(list_historical_data__daily))

    #
    # prepare data, and add calculated fields to data frame
    #
    df['date_str'] = df['date']
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df['price_close_lag'] = df['price_close'].shift(1)
    df['price_close_lead'] = df['price_close'].shift(-1)

    df.insert(0, 'date_id', range(1, 1 + len(df)))

    #
    # returns data frame
    #
    return df