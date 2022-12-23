# Generated by Django 4.1.3 on 2022-12-14 06:50

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0038_classificationmodel_results'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='experiment_type',
            field=models.CharField(blank=True, choices=[('input', 'Input'), ('columnformatchange', 'Column format change'), ('featureengineering', 'Feature engineering'), ('stationarity', 'Stationarity test'), ('manualvariableselection', 'Manual variable selection'), ('classificationmodel', 'Build logistic regression model')], default='input', max_length=100, null=True),
        ),
    ]