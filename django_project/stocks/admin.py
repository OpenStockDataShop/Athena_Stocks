from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import Fav_Stocks

#inspired by https://stackoverflow.com/questions/58272409/how-to-restrict-access-for-staff-users-to-see-only-their-information-in-django-a
class UserAdmin(BaseUserAdmin):
    def has_add_permission(self, request, obj=None):
        return request.user.is_superuser
    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser
    def has_change_permission(self, request, obj=None):
        return request.user.is_superuser
# Register your models here.
admin.site.register(Fav_Stocks)
