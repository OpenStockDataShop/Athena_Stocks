from django.urls import path
from . import views

#This was inspired by Corey Schaffer's tutorials as well as https://docs.djangoproject.com/en/3.2/intro/tutorial03/
urlpatterns = [
    path('', views.home, name="stocks-home"),
    path('about/', views.about, name="stocks-about"),
    path('UserPage/', views.userPage, name="stocks-UserPage"),
    path('Donate/', views.Donate, name="stocks-Donate"),
]