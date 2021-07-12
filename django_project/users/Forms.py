from django import forms
from django.forms.utils import to_current_timezone
from stocks import models as m
from django.forms import ModelForm
from django.contrib import admin
from datetime import datetime, timezone

class cFav_Stocks(forms.ModelForm):
    class Meta:
        model = m.Fav_Stocks
        fields = ('user', 'stocks', 'author')