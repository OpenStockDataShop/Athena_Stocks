# import libraries 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import date, timedelta
import pandas as pd

def get_momentum_recommendation(stock):
    """
        input: a single ticker
        output: predicted price for the stock
    """

    # import historical prices from yahoo finance 
    period1 = int(time.mktime((date.today()-timedelta(days=30)).timetuple()))
    period2 = int(time.mktime(date.today().timetuple()))
    interval = '1d' # 1wk, 1m
    query = f'https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query) # use yahoo finance historical prices as API
    print(df.head(5))

    # calculate 20-day average 
    closing_prices = df['Close'].to_list()
    twenty_day_average = sum(closing_prices[:-1])/len(closing_prices[:-1])
    print(closing_prices)

    rec = ''
    if closing_prices[-1] < twenty_day_average:
        rec = 'Buy'
    elif closing_prices[-1] > twenty_day_average:
        rec = 'Sell'
    else:
        rec = 'Hold'

    return (twenty_day_average, rec)