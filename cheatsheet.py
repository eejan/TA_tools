
import pandas as pd
import numpy as np

### tips on ipython notebook
# in notebook add:
# %load_ext autoreload
# %autoreload 1
# %aimport utils


# Using Pandas to calculate a 20-days span EMA. adjust=False specifies that we are interested in the recursive calculation mode.
data= pd.Series()
ema_short = data.ewm(span=20, adjust=False).mean()


# getting data
import yfinance as yf
from yahoofinancials import YahooFinancials
tsla_df = yf.download('TSLA', start='2019-01-01', end='2019-12-31', progress=False)


# get data using data reader
import pandas as pd
import numpy as np
from pandas_datareader import data
import datetime
tickers = ['AAPL', 'MSFT', 'NFLX', 'AMZN', 'GOOG']
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2018, 12, 31)
df = pd.DataFrame([data.DataReader(ticker, 'yahoo', start, end)['Adj Close'] for ticker in tickers]).T
df.columns = tickers


#getting data by quandl
import quandl
quandl.ApiConfig.api_key = '-dJhdMB1EcYEkxzxEGMA' # 'your_api_key_here'
stocks = ['AAPL','AMZN','GOOGL','FB']
data = quandl.get_table('WIKI/PRICES', ticker = stocks,
                        qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                        date = { 'gte': '2016-1-1', 'lte': '2017-12-31' }, paginate=True)
df = data.set_index('date')
table = df.pivot(columns='ticker')
table.columns = [col[1] for col in table.columns]


# re-index by different date range  ; but it may not work.. range from Sun - Sat..., which df has no data ..
df = pd.DataFrame()
all_weeks = pd.date_range(start='2019-01-03', end='2019-12-31', freq='W')
df = df.loc[all_weeks]


# rolling function  on moving average
df['mah15'] = df['Aju Close'].rolling(15, center=True, win_type='hamming').mean()


# return
#log return
log_returns = np.log(data).diff()

# simple return
returns = data.pct_change(1)