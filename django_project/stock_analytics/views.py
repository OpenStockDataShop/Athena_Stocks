from stocks.models import Fav_Stocks
from django.shortcuts import render
from django.http import HttpResponse
from .models import Stock

from .lstm import get_lstm_rec

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
    list_of_list = []
    for fav in fav_list:
        query_results = Stock.objects.filter(name=fav).order_by('date')
        price_list = []
        for s in query_results:
            price_list.append(s.price)
        list_of_list.append(price_list)


    # logic for moving average - momentum trading
    avg_list = []
    momentum_rec = []
    for list in list_of_list:
        avg = sum(list[:-1])/len(list)
        avg_list.append(avg)
        if list[-1] < avg:
            momentum_rec.append('buy')
        else:
            momentum_rec.append('sell')
    

    # logic for lstm 
    rec = get_lstm_rec()

    lstm_rec = [rec, rec]

    context = {
        'user': request.user,
        'momentum_rec': zip(fav_list, momentum_rec),
        'lstm_rec': zip(fav_list, lstm_rec)
    }

    return render(request, 'stocks/trading_rec.html', context)
