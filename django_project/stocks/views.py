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
    return render(request, 'stocks/about.html', context)

# <!-- https://docs.djangoproject.com/en/3.2/topics/db/queries/ inspiration taken from this website for fetching data from db -->


# grabbed second latest price inspired by the code from below
# https://stackoverflow.com/questions/35307405/how-to-get-second-last-record-in-a-queryset-in-django

def get_latest_prices(stock):

    # if stock is in database, get it from database
    target_stock_prices = Stock.objects.filter(name=stock).all().order_by('-date')
    num_records = len(target_stock_prices)
    if num_records >= 2:
        latest_record = target_stock_prices[0]
        second_to_latest_record = target_stock_prices[1]
        latest_price = latest_record.price
        second_to_latest_price = second_to_latest_record.price
        percent_change = (latest_price / second_to_latest_price - 1) * 100
        return latest_price, second_to_latest_price, percent_change
    # if stock is not in database, get it from web
    else:
        # import historical prices from yahoo finance 
        period1 = int(time.mktime((date.today()-timedelta(days=10)).timetuple()))
        period2 = int(time.mktime(date.today().timetuple()))
        interval = '1d' # 1wk, 1m
        query = f'https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
        df = pd.read_csv(query) # use yahoo finance historical prices as API
        dates = df['Date'].to_list()
        closing_prices = df['Close'].to_list()
        latest_price = closing_prices[-1]
        latest_date = dates[-1]
        # save it into the database
        z = Stock(name=stock, date=latest_date, price=latest_price)
        z.save()

        second_to_latest_price = closing_prices[-2]
        percent_change = (latest_price / second_to_latest_price -1) * 100

        return latest_price, second_to_latest_price, percent_change


# def get_latest_prices(stock):

#     # if stock is in database, get it from database
#     target_stock = Stock.objects.filter(name=stock)
#     if target_stock:
#         query_result = Stock.objects.filter(name=stock).latest('date')
#         price = query_result.price
#         return price
#     # if stock is not in database, get it from web
#     else:
#         # import historical prices from yahoo finance 
#         period1 = int(time.mktime((date.today()-timedelta(days=5)).timetuple()))
#         period2 = int(time.mktime(date.today().timetuple()))
#         interval = '1d' # 1wk, 1m
#         query = f'https://query1.finance.yahoo.com/v7/finance/download/{stock}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
#         df = pd.read_csv(query) # use yahoo finance historical prices as API
#         dates = df['Date'].to_list()
#         closing_prices = df['Close'].to_list()
#         print(dates, closing_prices)
#         price = closing_prices[-1]
#         latest_date = dates[-1]
#         # save it into the database
#         z = Stock(name=stock, date=latest_date, price=price)
#         z.save()

#         return price



@login_required
def userPage(request):
    if(request.user.is_authenticated):
        the_user = request.user.username
        stock_list = []
        comment_list = []
        temp = []
        temp = (Fav_Stocks.objects.all().values())
        for i in temp:
            if(the_user == i['user']):
                stock_list.append(i['stocks'].upper())

        # # only for testing and when I input wrong ticker
        # stock_prices = [120] * len(stock_list)

        # giving me 500 status code.. maybe hitting yahoo finance too many time?
        latest_prices = []
        second_latest_prices = []
        changes = []
        for stock in stock_list:
            latest_price, second_latest_price, change = get_latest_prices(stock)
            latest_prices.append(round(latest_price,2))
            second_latest_prices.append(round(second_latest_price, 2))
            changes.append(round(change, 2))

        # comment_list = []
        # date_created_list = []
        # for stock in stock_list:
        #         query_result = Fav_Stocks.objects.filter(stocks=stock).values()
        #         for q in query_result:
        #             if not q['comment']:
        #                 print("no comment")
        #                 comment_list.append("no comment")
        #             else:
        #                 comment_list.append(q['comment'])
        #             date_created_list.append(q['date_made'])

        # print(stock_list)
        # print(latest_prices)
        # print(comment_list)
        # print(date_created_list)

        context = {
            'the_user': the_user,
            'users_stocks': zip(stock_list, latest_prices, second_latest_prices, changes),
        }
    return render(request, 'stocks/UserPage.html', context)


def Donate(request):
    return render(request, 'stocks/Donate.html')
