# Generated by Django 4.1.5 on 2023-01-14 09:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0003_alter_classificationmodel_ignored_columns'),
    ]

    operations = [
        migrations.AddField(
            model_name='resultsclassificationmodel',
            name='intercept',
            field=models.TextField(blank=True, max_length=20000, null=True),
        ),
    ]