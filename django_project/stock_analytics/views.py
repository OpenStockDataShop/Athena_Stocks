from stocks.models import Fav_Stocks
from django.shortcuts import redirect, render
from django.http import HttpResponse
from .models import Stock

from .lstm import get_lstm_recommendation
from .momentum import get_momentum_recommendation
from .sentiment import getFullArticleTextFromURL, getSentimentFromText



# https://docs.djangoproject.com/en/3.2/ref/models/querysets/#queryset-api

# Create your views here.


def index(request):

    # if logged in, retrieve the user's favorite stocks list
    if (request.user.is_authenticated):
        the_user = request.user.username
        fav_stocks = Fav_Stocks.objects.filter(user=the_user)
        fav_list = []
        for fav in fav_stocks:
            fav_list.append(fav.stocks.upper())

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

        # sentiment of the latest news on the stock
        sentiment_scores = []
        for stock in fav_list:
            latest_news = getFullArticleTextFromURL(f'https://finance.yahoo.com/quote/{stock}/news?p={stock}')
            sentiment_list = getSentimentFromText(latest_news)
            sentiment_scores.append(sentiment_list[3])

        context = {
            'user': request.user,
            'momentum_rec': zip(fav_list, momentum_prices, momentum_recs),
            'lstm_rec': zip(fav_list, lstm_prices, lstm_recs), 
            'sentiment': zip(fav_list, sentiment_scores),
        }

        return render(request, 'stocks/analytics.html', context)

    else:

        return redirect('login/')
