# Generated by Django 4.1.3 on 2022-12-03 13:00

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0013_rename_experiment_id_variables_experiment_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='variables',
            old_name='var_id',
            new_name='variable_id',
        ),
        migrations.RenameField(
            model_name='variables',
            old_name='var_name',
            new_name='variable_name',
        ),
    ]
