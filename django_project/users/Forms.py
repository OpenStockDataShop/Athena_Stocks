from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.forms.utils import to_current_timezone
from stocks import models as m
from django.forms import ModelForm
from django.contrib import admin
from datetime import datetime, timezone

#inspired by https://www.geeksforgeeks.org/django-modelform-create-form-from-models/
class cFav_Stocks(forms.ModelForm):
    class Meta:
        model = m.Fav_Stocks
        fields = ('user', 'stocks', 'author')


# following this tutorial: https://www.notimedad.dev/customizing-django-builtin-login-form/
class Custom_Login_Form(AuthenticationForm):
    def _init_(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].widget.attrs.update({
            'class': 'my-username-class',
            'placeholder': 'Username',
        })
        self.fields['password'].widget.attrs.update({
            'class': 'my-password-class',
            'placeholder': 'Password',
        })
