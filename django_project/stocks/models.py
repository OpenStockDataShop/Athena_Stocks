from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

# Create your models here.
#inspired by https://docs.djangoproject.com/en/3.2/topics/db/models/
class Fav_Stocks(models.Model):
    user = models.CharField(max_length=100)
    stocks = models.TextField()
    date_made = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
