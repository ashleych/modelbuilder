# Generated by Django 4.1.3 on 2023-01-03 06:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0065_rename_precision_classificationmetrics_pr_thresholds_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='classificationmetrics',
            name='precision',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='classificationmetrics',
            name='recall',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
