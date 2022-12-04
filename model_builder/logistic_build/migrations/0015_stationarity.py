# Generated by Django 4.1.3 on 2022-12-03 18:49

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0014_rename_var_id_variables_variable_id_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='Stationarity',
            fields=[
                ('experiment_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='logistic_build.experiment')),
                ('do_adf', models.BooleanField(default=False)),
                ('do_kpss', models.BooleanField(default=False)),
                ('significance', models.FloatField(blank=True, default=0.05, null=True)),
                ('kpss_pass_vars', models.TextField(blank=True, max_length=20000, null=True)),
                ('adf_pass_vars', models.TextField(blank=True, max_length=20000, null=True)),
            ],
            bases=('logistic_build.experiment',),
        ),
    ]
