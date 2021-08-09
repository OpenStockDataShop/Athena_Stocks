from django.db import models

# Create your models here.
class Stock(models.Model):
    name = models.CharField(max_length=255)
    date = models.DateField()
    price = models.FloatField()

    def __str__(self):
        return self.name

# create stock_ticker model here
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





