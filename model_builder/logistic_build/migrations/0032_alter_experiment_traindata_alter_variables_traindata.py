# Generated by Django 4.1.3 on 2022-12-09 03:14

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0031_alter_experiment_experiment_type'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='traindata',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='input_train_data', to='logistic_build.traindata'),
        ),
        migrations.AlterField(
            model_name='variables',
            name='traindata',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='logistic_build.traindata'),
        ),
    ]