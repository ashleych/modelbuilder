# Generated by Django 4.1.3 on 2022-12-28 10:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0054_experiment_all_preceding_experiments'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='name',
            field=models.CharField(default='manual', max_length=100),
            preserve_default=False,
        ),
    ]