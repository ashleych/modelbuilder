# Generated by Django 4.1.5 on 2023-01-06 05:55

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0073_remove_resultsfeatureselection_featureselection_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='topmodels',
            old_name='result',
            new_name='results',
        ),
    ]