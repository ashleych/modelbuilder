# Generated by Django 4.1.3 on 2022-12-14 06:32

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0037_manualvariableselection_keep_columns'),
    ]

    operations = [
        migrations.CreateModel(
            name='Classificationmodel',
            fields=[
                ('experiment_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='logistic_build.experiment')),
                ('label_col', models.CharField(blank=True, max_length=100, null=True)),
                ('train_split', models.FloatField(blank=True, null=True)),
                ('test_split', models.FloatField(blank=True, null=True)),
                ('feature_cols', models.TextField(blank=True, max_length=20000, null=True)),
                ('ignored_columns', models.TextField(blank=True, max_length=20000, null=True)),
                ('cross_validation', models.BooleanField(default=False)),
            ],
            bases=('logistic_build.experiment',),
        ),
        migrations.CreateModel(
            name='Results',
            fields=[
                ('experiment_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='logistic_build.experiment')),
            ],
            bases=('logistic_build.experiment',),
        ),
    ]