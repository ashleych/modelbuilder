# Generated by Django 4.1.3 on 2022-12-02 09:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Traindata',
            fields=[
                ('file_id', models.AutoField(primary_key=True, serialize=False)),
                ('train_path', models.CharField(max_length=50)),
                ('colnames', models.CharField(max_length=54)),
                ('relevant_col_names', models.CharField(max_length=500)),
            ],
        ),
    ]
