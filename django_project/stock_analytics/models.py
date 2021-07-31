from django.db import models

# Create your models here.
class Stock(models.Model):
    name = models.CharField(max_length=255)
    date = models.DateField()
    price = models.FloatField()

    def __str__(self):
        return self.name


