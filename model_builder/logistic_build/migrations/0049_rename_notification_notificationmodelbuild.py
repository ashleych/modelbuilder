# Generated by Django 4.1.3 on 2022-12-22 07:40

from django.conf import settings
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('logistic_build', '0048_alter_notification_managers_notification_created_by_and_more'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Notification',
            new_name='NotificationModelBuild',
        ),
    ]
