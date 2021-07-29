from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from .Forms import cFav_Stocks
from stocks import models as m
from django.utils import timezone
from datetime import datetime


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('/')
        else:
            print(form.errors)
            return redirect('/register')
    form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

# Create your views here.


def logout_view(request):
    if request.method == 'POST':
        logout(request)
        return redirect('/')
    return render(request, 'registration/logout.html')


def user_view(request):
    if request.method == 'POST':
        if(request.POST.get('send') == "delete"):
            stock = request.POST.get('stock')
            m.Fav_Stocks.objects.get(stocks=stock).delete()
            return redirect('/')

        username = request.POST.get('username')
        stock = request.POST.get('stock')
        author = request.user
        date_made = datetime.now()
        if(username != "" and stock != ""):
            z = m.Fav_Stocks(user=username, stocks=stock,
                             date_made=date_made, author=author)
            z.save()
            return redirect('/')
    form = cFav_Stocks
    return render(request, 'registration/the_stocks.html', {'form': form})
