import pandas as pd
from models import Stock

# idea to execute code when django starts up
# adopted from the below link: 
# https://stackoverflow.com/questions/2781383/where-to-put-django-startup-code
def populate_stock_db():
    nasdaq = pd.read_csv('static/nasdaq.csv')
    nyse = pd.read_csv('static/nyse.csv')
    etf = pd.read_csv('static/etf.csv')

    
    nasdaq_stocks = nasdaq[["Symbol", "Name", "Sector", "Industry"]].values.tolist()
    for stock in nasdaq_stocks:
        s = Stock(symbol=stock[0], name=stock[1], sector=stock[2], industry=[3])
        s.save()
    
    nyse_stocks = nyse[["Symbol", "Name", "Sector", "Industry"]].values.tolist()
    for stock in nyse_stocks:
        s = Stock(symbol=stock[0], name=stock[1], sector=stock[2], industry=[3])
        s.save()

    etf_stocks = etf[["Symbol", "Name"]].values.tolist()
    for etf in etf_stocks:
        s = Stock(symbol=etf[0], name=etf[1], sector="ETF", industry="ETF")
        s.save()

populate_stock_db()