from stocks.models import Fav_Stocks
from django.shortcuts import render
from django.http import HttpResponse
from .models import Stock


import math

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

    # Query historical prices of the favorite stock
    list_of_list = []
    for fav in fav_list:

        query_results = Stock.objects.filter(name='tsla').order_by('date')
        price_list = []
        for s in query_results:
            price_list.append(s.price)
        list_of_list.append(price_list)


    recommendations = ['buy']

    context = {
        'fav_stocks' : fav_list,
        'list_of_price_lists': list_of_list,
        'recommendations': recommendations,
    }

    return render(request, 'stock_analytics/home.html', context)
