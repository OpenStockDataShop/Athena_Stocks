from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="stocks-home"),
    path('about/', views.about, name="stocks-about"),
    path('UserPage/', views.userPage, name="stocks-UserPage"),
    path('Donate/', views.Donate, name="stocks-Donate"),
]