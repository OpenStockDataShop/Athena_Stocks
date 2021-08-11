from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from .Forms import cFav_Stocks
from stocks import models as m
from django.utils import timezone
from datetime import datetime

# placeholder code based on https://stackoverflow.com/questions/13523286/how-to-add-placeholder-to-forms-of-django-registration


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


def logout_view(request):
    logout(request)
    request.user = None
    return render(request, 'stocks/home.html')


def user_view(request, the_user):
    if request.method == 'POST':

        username = request.POST.get('username')
        stock = request.POST.get('stock').lower()
        author = request.user
        date_made = datetime.now()
        if(username != "" and stock != ""):
            z = m.Fav_Stocks(user=username, stocks=stock,
                             date_made=date_made, author=author)
            z.save()
            return redirect('/UserPage')
    
    form = cFav_Stocks

    context = {
        'form': form,
        'the_user': the_user, 
    }
    
    return render(request, 'registration/the_stocks.html', context)

def delete(request, the_user, stock):
    stock = stock.lower()
    query_result = m.Fav_Stocks.objects.filter(user=the_user).get(stocks=stock)
    query_result.delete()
    
    return redirect('/UserPage')

