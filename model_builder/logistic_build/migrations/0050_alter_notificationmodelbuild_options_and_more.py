# Generated by Django 4.1.3 on 2022-12-23 07:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('logistic_build', '0049_rename_notification_notificationmodelbuild'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='notificationmodelbuild',
            options={'ordering': ['-timestamp']},
        ),
        migrations.AlterModelManagers(
            name='notificationmodelbuild',
            managers=[
            ],
        ),
        migrations.AddField(
            model_name='notificationmodelbuild',
            name='experiment_type',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
