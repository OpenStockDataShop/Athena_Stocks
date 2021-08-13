"""django_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from django.contrib.auth.views import LoginView
from users import views as user_views
from users.Forms import Custom_Login_Form

#https://www.youtube.com/watch?v=q4jPR-M0TAQ We learned how to include urlpatterns from Corey Schafer please check out his videos
urlpatterns = [
    path('admin/', admin.site.urls, name='site-admin'),
    path('register/', user_views.register, name='register'),
    path('login/', LoginView.as_view(
        authentication_form=Custom_Login_Form),
        name='login'),
    path('enter_data/<slug:the_user>', user_views.user_view, name='the_user_view'),
    path('delete_data/<slug:the_user>/<slug:stock>', user_views.delete, name='delete_stock'),
    path('logout/', user_views.logout_view, name='logout'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('analytics/', include('stock_analytics.urls'), name='analytics'),
    path('', include('stocks.urls')),
]
