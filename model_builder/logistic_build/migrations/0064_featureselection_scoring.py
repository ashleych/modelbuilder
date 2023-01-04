# Generated by Django 4.1.3 on 2023-01-01 06:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0063_rename_ignored_features_featureselection_exclude_features_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='featureselection',
            name='scoring',
            field=models.CharField(choices=[('roc_auc', 'ROC AUC'), ('mse', 'MSE')], default='roc_auc', max_length=10),
        ),
    ]
