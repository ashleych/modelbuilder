# Generated by Django 4.1.3 on 2022-12-17 16:03

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0040_experiment_artefacts_directory_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ClassificationMetrics',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('type', models.CharField(blank=True, max_length=100, null=True)),
                ('FPR', models.TextField(blank=True, max_length=20000, null=True)),
                ('TPR', models.TextField(blank=True, max_length=20000, null=True)),
                ('areaUnderROC', models.FloatField(blank=True, null=True)),
                ('precision', models.TextField(blank=True, max_length=20000, null=True)),
                ('recall', models.TextField(blank=True, max_length=20000, null=True)),
                ('thresholds', models.TextField(blank=True, max_length=20000, null=True)),
                ('areaUnderPR', models.FloatField(blank=True, null=True)),
            ],
        ),
        migrations.DeleteModel(
            name='Results',
        ),
        migrations.AddField(
            model_name='classificationmodel',
            name='test_results',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='test_results', to='logistic_build.classificationmetrics'),
        ),
        migrations.AddField(
            model_name='classificationmodel',
            name='train_results',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='train_results', to='logistic_build.classificationmetrics'),
        ),
    ]
