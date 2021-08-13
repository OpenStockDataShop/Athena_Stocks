from django.apps import AppConfig

class StockAnalyticsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stock_analytics'

    def ready(self):
        
        import pandas as pd
        from .models import Stock
        import os 

        nasdaq_file_path = os.path.join(os.path.dirname(__file__), 'static/nasdaq.csv')
        nyse_file_path = os.path.join(os.path.dirname(__file__), 'static/nyse.csv')
        etf_file_path = os.path.join(os.path.dirname(__file__), 'static/etf.csv')

        nasdaq = pd.read_csv(nasdaq_file_path)
        nyse = pd.read_csv(nyse_file_path)
        etf = pd.read_csv(etf_file_path)

        nasdaq_stocks = nasdaq[["Symbol", "Name", "Sector", "Industry"]].values.tolist()
        for stock in nasdaq_stocks:
            if not Stock.objects.filter(symbol=stock[0]):
                s = Stock(symbol=stock[0], name=stock[1], sector=stock[2], industry=[3])
                s.save()


        nyse_stocks = nyse[["Symbol", "Name", "Sector", "Industry"]].values.tolist()
        for stock in nyse_stocks:
            if not Stock.objects.filter(symbol=stock[0]):
                s = Stock(symbol=stock[0], name=stock[1], sector=stock[2], industry=[3])
                s.save()

        etf_stocks = etf[["SYMBOL", "NAME"]].values.tolist()
        for etf in etf_stocks:
            if not Stock.objects.filter(symbol=stock[0]):
                s = Stock(symbol=etf[0], name=etf[1], sector="ETF", industry="ETF")
                s.save()
