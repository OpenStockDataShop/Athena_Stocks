# Generated by Django 3.2.5 on 2021-07-30 20:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stocks', '0002_auto_20210730_1814'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fav_stocks',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
    ]