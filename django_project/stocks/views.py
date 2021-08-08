from django.shortcuts import render
from .models import Fav_Stocks
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
import time
from datetime import date, timedelta
import pandas as pd
from stock_analytics.models import Stock
# Create your views here.


def home(request):
    context = {
        'user': request.user.username,
        'fav_stocks': Fav_Stocks.objects.all()
    }
    return render(request, 'stocks/home.html', context)


def about(request):
    context = {
        'user': request.user.username
    }
    print(context)
    return render(request, 'stocks/about.html', context)

# <!-- https://docs.djangoproject.com/en/3.2/topics/db/queries/ inspiration taken from this website for fetching data from db -->

def get_latest_price(stock):


    # if stock is in database, get it from database
    target_stock = Stock.objects.filter(name=stock)
    if target_stock:
        query_result = Stock.objects.filter(name=stock).latest('date')
        price = query_result.price
        return price
    # if stock is not in database, get it from web
    else:
        # import historical prices from yahoo finance 
        period1 = int(time.mktime((date.today()-timedelta(days=1)).timetuple()))
        period2 = int(time.mktime(date.today().timetuple()))
        interval = '1d' # 1wk, 1m
        query = f'https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
        df = pd.read_csv(query) # use yahoo finance historical prices as API
        dates = df['Date'].to_list()
        closing_prices = df['Close'].to_list()
        price = closing_prices[-1]
        latest_date = dates[-1]
        # save it into the database
        z = Stock(name=stock, date=latest_date, price=price)
        z.save()

        return price


@login_required
def userPage(request):
    if(request.user.is_authenticated):
        the_user = request.user.username
        stock_list = []
        temp = []
        temp = (Fav_Stocks.objects.all().values())
        for i in temp:
            print(i['user'])
            if(the_user == i['user']):
                stock_list.append(i['stocks'].upper())
        print(stock_list)

        # giving me 500 status code.. maybe hitting yahoo finance too many time?
        stock_prices = []
        for stock in stock_list:
            price = get_latest_price(stock)
            stock_prices.append(round(price,2))

        context = {
            'the_user': the_user,
            'users_stocks': zip(stock_list, stock_prices),
        }
        print("context = ", context)
    return render(request, 'stocks/UserPage.html', context)


def Donate(request):
    return render(request, 'stocks/Donate.html')
