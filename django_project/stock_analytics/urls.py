from django.urls import path
from django.contrib.auth.views import LoginView
from . import views
from users.Forms import Custom_Login_Form

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', LoginView.as_view(
        authentication_form=Custom_Login_Form),
        name='login'),
]
