# Generated by Django 4.1.3 on 2022-12-24 03:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0051_experiment_run_in_the_background'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='lock_now',
            field=models.BooleanField(default=False),
        ),
    ]