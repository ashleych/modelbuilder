# Generated by Django 4.1.3 on 2022-12-07 05:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0025_rename_fibonacci_fibonnaci'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='experiment_type',
            field=models.CharField(blank=True, choices=[('Input', 'Input'), ('Column_format_change', 'Column format change'), ('Feature_engineering', 'Feature engineering'), ('Stationarity_test', 'Stationarity test'), ('Manual_variable_selection', 'Manual variable selection')], default='Input', max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='fibonnaci',
            name='number',
            field=models.PositiveSmallIntegerField(default=5),
        ),
    ]