from django.contrib import admin
from .models import Stock, StockPrices, StockNews

# Register your models here.
admin.site.register(Stock)
admin.site.register(StockPrices)
admin.site.register(StockNews)
