import requests
import pandas as pd


class AlphaVantage:

    def __init__(self, symbol_code, interval="60min", outputsize="compact"):
        self.base_url = "https://www.alphavantage.co"
        self.default_endpoint = "/query"
        self.api_key = "<enter your API Key here>"
        self.symbol_code = symbol_code
        self.interval = interval
        self.outputsize = outputsize

    def intraday(self):
        parameters = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": self.symbol_code,
            "interval": self.interval,
            "outputsize": self.outputsize,
            "datatype": "json",
            "apikey": self.api_key
        }

        response_intraday = requests.get("{0}{1}".format(self.base_url, self.default_endpoint), params=parameters, stream=True).json()

        return response_intraday


    def daily(self):
        parameters = {
            "function": "TIME_SERIES_DAILY",
            "symbol": self.symbol_code,
            "outputsize": self.outputsize,
            "datatype": "json",
            "apikey": self.api_key
        }

        response_daily = requests.get("{0}{1}".format(self.base_url, self.default_endpoint), params=parameters, stream=True).json()

        return response_daily
    