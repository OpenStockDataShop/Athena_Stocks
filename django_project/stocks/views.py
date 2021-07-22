from django.shortcuts import render
from .models import Fav_Stocks
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from users import models as m1

# Create your views here.

def home(request):
    context = {
        'fav_stocks': Fav_Stocks.objects.all()
    }
    return render(request, 'stocks/home.html', context)

def about(request):
    return render(request, 'stocks/about.html')

<!-- https://docs.djangoproject.com/en/3.2/topics/db/queries/ inspiration taken from this website for fetching data from db -->
@login_required
def userPage(request):
    if(request.user.is_authenticated):
        the_user = request.user.username
        stock_list = []
        temp = []
        temp = (Fav_Stocks.objects.all().values())
        for i in temp:
            print(i['user'])
            if(the_user==i['user']):
                stock_list.append(i['stocks'])
        print(stock_list)
        context = {
            'the_user': the_user,
            'users_stocks': stock_list,
        }
        print("context = ", context)
    return render(request, 'stocks/UserPage.html', context)

def Donate(request):
    return render(request, 'stocks/Donate.html')
