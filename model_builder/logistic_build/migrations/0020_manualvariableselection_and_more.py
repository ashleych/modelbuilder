# Generated by Django 4.1.3 on 2022-12-04 15:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0019_alter_experiment_output_data'),
    ]

    operations = [
        migrations.CreateModel(
            name='Manualvariableselection',
            fields=[
                ('experiment_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='logistic_build.experiment')),
            ],
            bases=('logistic_build.experiment',),
        ),
        migrations.AlterField(
            model_name='experiment',
            name='experiment_type',
            field=models.CharField(blank=True, choices=[('Input', 'Input'), ('Column_format_change', 'Column_format_change'), ('Feature_engineering', 'Feature_engineering'), ('Stationarity_test', 'Stationarity_test'), ('Manual_variable_selection', 'Manual_variable_selection')], max_length=100, null=True),
        ),
    ]
