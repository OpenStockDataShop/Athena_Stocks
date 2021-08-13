from django.apps import AppConfig
# from .models import Stock

# idea to execute code when django starts up
# adopted from the below link: 
# https://pythonin1minute.com/where-to-put-django-startup-code/
class StockAnalyticsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stock_analytics'

    # def ready(self): 
    #     # startup code here
    #     nasdaq_query = 
