from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from .Forms import cFav_Stocks
from stocks import models as m
from django.utils import timezone
from datetime import datetime

from django.contrib import messages
import pandas as pd
import os
import sys

# placeholder code based on https://stackoverflow.com/questions/13523286/how-to-add-placeholder-to-forms-of-django-registration

#https://stackoverflow.com/questions/35602049/how-to-insert-data-to-django-database-from-views-py-file and https://stackoverflow.com/questions/55782147/how-can-i-send-data-to-a-database-from-a-view-in-django
def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        form.fields['username'].widget.attrs.update({
            'placeholder': 'Username'
        })
        form.fields['password1'].widget.attrs.update({
            'placeholder': 'Password',
        })
        form.fields['password2'].widget.attrs.update({
            'placeholder': 'Password Confirmation'
        })
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            context = {
                'the_user': username,
                'users_stocks': zip([], [], [], []),
            }
            return render(request, 'stocks/UserPage.html', context)
        else:
            print(form.errors)
            return render(request, 'registration/register.html', {'form': form})
    form = UserCreationForm()
    form.fields['username'].widget.attrs.update({
        'placeholder': 'Username'
    })
    form.fields['password1'].widget.attrs.update({
        'placeholder': 'Password'
    })
    form.fields['password2'].widget.attrs.update({
        'placeholder': 'Password Confirmation'
    })

    context = {
        'user': request.user.username,
        'form': form
    }

    return render(request, 'registration/register.html', context)

#inspired by https://stackoverflow.com/questions/25251719/how-can-i-logout-a-user-in-django
def logout_view(request):
    logout(request)
    request.user = None
    return render(request, 'stocks/home.html')


def user_view(request, the_user):
    if request.method == 'POST':

        projdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/').replace('C:', '')
        datadir = projdir.replace('\\', '/') + '/stock_analytics/data'
        csvpath = datadir.replace('\\','/') + '/nasdaq_screener_ticker_database.csv'
 
        if os.path.exists(csvpath):
            dataset = pd.read_csv(csvpath)
        else:
            dataset = pd.DataFrame()

        username = request.POST.get('username')
        stock = request.POST.get('stock').lower()
        author = request.user
        date_made = datetime.now()

        data = dict()

        valid_stock = True
        if(username != "" and stock != ""):
            if (dataset.shape[0] > 0):
                valid_stock = False
                if stock in dataset['Symbol'].str.lower().tolist():
                    valid_stock = True
                else:
                     messages.error(request, "Entered stock = " + stock + " is not a valid stock symbol")
            if valid_stock:   
                query_result = m.Fav_Stocks.objects.filter(user=the_user,stocks=stock)
                if len(query_result) > 0:
                    messages.info(request, "Entered stock = " + stock + " already exists in your favorite stocks.")
                else:
                    z = m.Fav_Stocks(user=username, stocks=stock,
                             date_made=date_made, author=author)
                    z.save()
            return redirect('/UserPage',data)
    
    form = cFav_Stocks

    context = {
        'form': form,
        'the_user': the_user, 
    }
    
    return render(request, 'registration/the_stocks.html', context)

#inspired by https://stackoverflow.com/questions/3805958/how-to-delete-a-record-in-django-models and https://stackoverflow.com/questions/57842727/how-to-select-only-one-row-for-one-user-in-django
def delete(request, the_user, stock):
    stock = stock.lower()
    query_result = m.Fav_Stocks.objects.filter(user=the_user,stocks=stock)
    if len(query_result) > 0:
        query_result.delete()
    
    return redirect('/UserPage')

