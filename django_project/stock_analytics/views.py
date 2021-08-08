from stocks.models import Fav_Stocks
from django.shortcuts import redirect, render
from django.http import HttpResponse
from .models import Stock

from .lstm import get_lstm_recommendation
from .momentum import get_momentum_recommendation

# https://docs.djangoproject.com/en/3.2/ref/models/querysets/#queryset-api

# Create your views here.


def index(request):

    # if logged in, retrieve the user's favorite stocks list
    if (request.user.is_authenticated):
        the_user = request.user.username
        fav_stocks = Fav_Stocks.objects.filter(user=the_user)
        fav_list = []
        for fav in fav_stocks:
            fav_list.append(fav.stocks)

        # query historical prices of the favorite stock
        # list_of_list = []
        # for fav in fav_list:
        #     query_results = Stock.objects.filter(name=fav).order_by('date')
        #     price_list = []
        #     for s in query_results:
        #         price_list.append(s.price)
        #     list_of_list.append(price_list)



        # logic for 20 moving average - momentum trading
        momentum_prices = []
        momentum_recs = []
        for stock in fav_list:
            rec_tuple = get_momentum_recommendation(stock)
            print(rec_tuple)
            momentum_prices.append(rec_tuple[0])
            momentum_recs.append(rec_tuple[1])

        # logic for lstm
        lstm_prices = []
        lstm_recs = []
        for stock in fav_list:
            rec = get_lstm_recommendation(stock)
            lstm_prices.append(rec[0])
            lstm_recs.append(rec[1])


        # only for testing - faster testing
        # momentum_prices = [120] * len(fav_list)
        # momentum_recs = [ 'Sell'] * len(fav_list)
        # lstm_prices = [120] * len(fav_list)
        # lstm_recs = ['Buy'] * len(fav_list)

        context = {
            'user': request.user,
            'momentum_rec': zip(fav_list, momentum_prices, momentum_recs),
            'lstm_rec': zip(fav_list, lstm_prices, lstm_recs)
        }

        return render(request, 'stocks/analytics.html', context)

    else:

        return redirect('login/')
