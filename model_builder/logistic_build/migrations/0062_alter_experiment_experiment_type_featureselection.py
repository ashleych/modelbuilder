# Generated by Django 4.1.3 on 2022-12-31 12:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0061_alter_experiment_experiment_type'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='experiment_type',
            field=models.CharField(blank=True, choices=[('input', 'Input'), ('columnformatchange', 'Column format change'), ('featureengineering', 'Feature engineering'), ('stationarity', 'Stationarity test'), ('manualvariableselection', 'Manual variable selection'), ('featureselection', 'Feature Selection'), ('classificationmodel', 'Build logistic regression model'), ('regressionmodel', 'Build regression model')], default='input', max_length=100, null=True),
        ),
        migrations.CreateModel(
            name='Featureselection',
            fields=[
                ('experiment_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='logistic_build.experiment')),
                ('label_col', models.CharField(blank=True, max_length=100, null=True)),
                ('feature_cols', models.TextField(blank=True, max_length=20000, null=True)),
                ('train_split', models.FloatField(blank=True, null=True)),
                ('test_split', models.FloatField(blank=True, null=True)),
                ('ignored_features', models.TextField(blank=True, max_length=20000, null=True)),
                ('fixed_features', models.TextField(blank=True, max_length=20000, null=True)),
                ('cross_validation', models.BooleanField(default=False)),
                ('max_features', models.FloatField(blank=True, null=True)),
                ('min_features', models.FloatField(blank=True, null=True)),
                ('short_list_max_features', models.FloatField(blank=True, null=True)),
                ('regression_or_classification', models.CharField(choices=[('regression', 'Regression'), ('classification', 'Classification')], default='regression', max_length=20)),
                ('remove_constant_features', models.BooleanField(blank=True, default=False, null=True)),
                ('remove_quasi_constant_features', models.BooleanField(blank=True, default=False, null=True)),
                ('variance_threshold', models.BooleanField(blank=True, default=False, null=True)),
                ('correlation_check', models.BooleanField(blank=True, default=False, null=True)),
                ('correlation_threshold', models.FloatField(blank=True, null=True)),
                ('treat_missing', models.BooleanField(blank=True, default=False, null=True)),
                ('variables_selected', models.BooleanField(blank=True, default=False, null=True)),
                ('do_exhaustive_search', models.BooleanField(blank=True, default=False, null=True)),
                ('results', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='logistic_build.resultsregressionmodel')),
            ],
            bases=('logistic_build.experiment',),
        ),
    ]
