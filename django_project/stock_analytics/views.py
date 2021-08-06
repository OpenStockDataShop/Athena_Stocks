from stocks.models import Fav_Stocks
from django.shortcuts import render
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
        momentum_recs = []
        for stock in fav_list:
            rec_tuple = get_momentum_recommendation(stock)
            momentum_recs.append(rec_tuple)
    

        # logic for lstm 
        lstm_recs = []
        for stock in fav_list:
            rec = get_lstm_recommendation(stock)
            lstm_recs.append(rec)
        print(lstm_recs)
        

        context = {
            'user': request.user,
            'momentum_rec': zip(fav_list, momentum_recs),
            'lstm_rec': zip(fav_list, lstm_recs)
        }

        return render(request, 'stocks/trading_rec.html', context)
    
    else:

        return HttpResponse('You are not logged in. Please login or register.')

