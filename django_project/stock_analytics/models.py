from django.db import models
from django.db.models.deletion import PROTECT
from django.db.models.fields.related import ForeignKey
from datetime import date

# Create your models here.
#learned from Django Documentation
class Stock(models.Model):

    class Meta:
        db_table = 'Stock'

    symbol = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    sector = models.CharField(max_length=255)
    industry = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class StockPrices(models.Model):

    class Meta:
        db_table = 'StockPrices'

    symbol = models.CharField(max_length=255)
    date = models.DateField()
    price = models.FloatField()

# Model for historical stock data including  sentiment scores
class StockNews(models.Model):
    name = models.CharField(max_length=255)
    date = models.DateField()
    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()
    adjclose = models.FloatField()
    volume = models.IntegerField()
    neg = models.FloatField()
    neu = models.FloatField()
    pos = models.FloatField()
    compound = models.FloatField()

    def __str__(self):
        return self.name
# create stock_ticker model here






