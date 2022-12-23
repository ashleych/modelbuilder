# Generated by Django 4.1.3 on 2022-12-18 06:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0041_classificationmetrics_delete_results_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='classificationmodel',
            name='test_results',
        ),
        migrations.RemoveField(
            model_name='classificationmodel',
            name='train_results',
        ),
        migrations.AddField(
            model_name='classificationmetrics',
            name='experiment_id',
            field=models.IntegerField(blank=True, null=True),
        ),
        migrations.CreateModel(
            name='ClassificationResults',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('coefficients', models.TextField(blank=True, max_length=20000, null=True)),
                ('feature_cols', models.TextField(blank=True, max_length=20000, null=True)),
                ('test_results', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='test_results', to='logistic_build.classificationmetrics')),
                ('train_results', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='train_results', to='logistic_build.classificationmetrics')),
            ],
        ),
        migrations.AddField(
            model_name='classificationmodel',
            name='results',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='logistic_build.classificationresults'),
        ),
    ]
