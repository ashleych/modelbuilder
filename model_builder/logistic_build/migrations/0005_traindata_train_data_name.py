# Generated by Django 4.1.3 on 2022-12-03 05:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0004_variables_var_name'),
    ]

    operations = [
        migrations.AddField(
            model_name='traindata',
            name='train_data_name',
            field=models.CharField(default='macro_data_1', max_length=100, unique=True),
        ),
    ]