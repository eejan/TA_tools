import quandl
import pandas as pd
import numpy as np
from pandas_datareader import data
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
from scipy.stats import linregress


def ATR(DF, n, close_column='Adj Close', high_column='High', low_column='Low'):   # average true range
    df = DF.copy()
    df['H-L'] = abs(df[high_column] - df[low_column])
    df['H-PC'] = abs(df[high_column] - df[close_column].shift(1))
    df['L-PC'] = abs(df[low_column] - df[close_column].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis = 1, skipna = False)
    df['ATR'] = df['TR'].rolling(n).mean()
    df = df.drop(['H-L', 'H-PC', 'L-PC'], axis=1)
    return df

# slope
def slope(ser, n):
    """
    function to calcualte slope using regression for n consecutive points
    """
    ser = (ser - ser.min()) / (ser.max() - ser.min())
    x = np.array(range(len(ser)))
    x = (x - x.min()) / (x.max() - x.min())
    slopes = [i * 0 for i in range(n - 1)]
    for i in range(n, len(ser) + 1):
        y_scaled = ser[i - n:i]
        x_scaled = x[i - n:i]
        reg = linregress(x=x_scaled, y=y_scaled)
        slopes.append(reg[0])

    #        x_scaled = sm.add_constant(x_scaled)
    #        model = sm.OLS(y_scaled,x_scaled)
    #        results = model.fit()
    #        slopes.append(results.params[-1])
    ##    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    ##    return np.array(slope_angle)
    return slopes


# on balance volume
def OBV(DF, close_column='Adj Close', vol_name='Volume'):
    """function to calculate On Balance Volume"""
    df = DF.copy()
    df['daily_ret'] = df[close_column].pct_change()
    df['direction'] = np.where(df['daily_ret']>=0,1,-1)
    df.iloc[0, df.columns.get_loc('direction')] = 0
#    df['direction'][0] = 0
    df['vol_adj'] = df[vol_name] * df['direction']
    df['obv'] = df['vol_adj'].cumsum()
    return df['obv']



# compounded annual growth return
def CAGR(DF, column_name='Adj Close'):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF[column_name].pct_change()
    df["cum_return"] = (1 + df["daily_ret"]).cumprod() -1   #modified by EE
    n = len(df)/252
    CAGR = (df["cum_return"][-1] + 1)**(1/n) - 1   # modified by EE
    return CAGR



def volatility(DF, colume_name='Adj Close',  ddof=1):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF[colume_name].pct_change()
    vol = df["daily_ret"].std(ddof=ddof) * np.sqrt(252)
    return vol


## this function may not be correct.. need additional verification  # 06/30
def max_dd(DF, colume_name="Adj Close"):
    "function to calculate max drawdown"
    df = DF.copy()
    df["daily_ret"] = DF[colume_name].pct_change()
    df["cum_return"] = (1 + df["daily_ret"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"] / df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd


def calmar(DF):
    "function to calculate calmar ratio"
    df = DF.copy()
    clmr = CAGR(df) / max_dd(df)
    return clmr



def sharpe(DF, rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf) / volatility(df)
    return sr


def sortino(DF, rf, tr=0, column_name='Adj Close', ddof=1):
    """function to calculate sortino ratio ; rf is the risk free rate; tr: target rate, usually=0, or risk free rate
    std, use ddof=1  , sample std.. not population std
    """

    df = DF.copy()
    df["daily_ret"] = DF[column_name].pct_change()

    # df["neg_ret"] = np.where(df["daily_ret"] < 0, df["daily_ret"], 0)
    # neg_vol = df["neg_ret"].std() * np.sqrt(252)
    neg_vol = df[df['daily_ret'] < tr].std(ddof=ddof) * np.sqrt(252)   # can we do 252?   , should we do positive ret = 0

    sr = (CAGR(df) - rf) / neg_vol
    return sr





## RSI :
# input is a dataFrame, output: add rsi in the dataframe
def RSI(stock, column="Adj Close", period=14):
    # Wilder's RSI
    close = stock[column]
    delta = close.diff()
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the exponential moving averages (EWMA)
    roll_up = up.ewm(com=period - 1, adjust=False).mean()
    roll_down = down.ewm(com=period - 1, adjust=False).mean().abs()

    # Calculate RS based on exponential moving average (EWMA)
    rs = roll_up / roll_down  # relative strength =  average gain/average loss

    rsi = 100 - (100 / (1 + rs))
    stock['RSI'] = rsi

    return stock



#import quandl
def get_quandl_data(stocksList=['AAPL','AMZN','GOOGL','FB'], start_date='2017-01-01', end_date='2018-01-01'):
    with open('quandl_api.txt', 'r') as inf:
        apikey = inf.readline().strip()
    print('apikey=', apikey, 'now use file input')
    quandl.ApiConfig.api_key = apikey #  '-dJhdMB1EcYEkxzxEGMA' # 'your_api_key_here'
    # stocks = ['AAPL','AMZN','GOOGL','FB']
    stocks = stocksList
    data = quandl.get_table('WIKI/PRICES', ticker = stocks,
                            qopts = { 'columns': ['date', 'ticker', 'adj_close'] },
                            date = { 'gte': start_date, 'lte': end_date }, paginate=True)
    df = data.set_index('date')
    table = df.pivot(columns='ticker')
    table.columns = [col[1] for col in table.columns]
    return table


# import pandas as pd
# from pandas_datareader import data
def get_yahoo_data_from_reader(tickerList=['AAPL', 'MSFT', 'NFLX', 'AMZN', 'GOOG'],
                               start_date='2020-01-01', end_date='2020-06-01'):
    tickers = tickerList
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    df = pd.DataFrame([data.DataReader(ticker, 'yahoo', start, end)['Adj Close'] for ticker in tickers]).T
    df.columns = tickers
    return df


def get_yahoo_data_1_ticker(ticker='IBM', start_date='2019-01-01', end_date='2020-06-06'):
    df = yf.download(ticker,
                      start=start_date,
                      end=end_date,
                      progress=False)
    return df


def get_yahoo_Adj_Close_tickerList(ticker=['IBM', 'MSFT'], start_date='2019-01-01', end_date='2020-06-06'):
    df = yf.download(ticker,
                      start=start_date,
                      end=end_date,
                      progress=False)['Adj Close']
    return df


