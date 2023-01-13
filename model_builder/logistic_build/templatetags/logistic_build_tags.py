# from django import template
from ..models import NotificationModelBuild

# register= template.Library()

from django.utils import timezone
def time_differencer(t2):
    t=timezone.now()- t2
    hours=round(t.seconds/60/60)
    mins=round(t.seconds/60)
    days= round(hours/24)
    if days:
        return str(days)+" days"
    if hours:
        return str(hours) +" hours"
    else:
        return str(mins) + " mins"


import datetime
from django import template

register = template.Library()

@register.simple_tag
def current_time(format_string):
    return datetime.datetime.now().strftime(format_string)


@register.simple_tag
def total_notification(name='total_notification'):
    time_since_creation_new=[]
    time_since_creation_old=[]
    number_of_notifications = 3
    new_notifications=None
    old_notifications=None
    # return NotificationModelBuild.objects.count()
    notifications=NotificationModelBuild.objects.all()[:number_of_notifications]
    new_notifications=NotificationModelBuild.objects.filter(is_read=False)[:number_of_notifications]
    count_new=new_notifications.count()
    if count_new<3:
        old_notifications=NotificationModelBuild.objects.filter(is_read=True)[:number_of_notifications-count_new]
        if old_notifications:
            for n in old_notifications:
                time_since_creation_old.append(time_differencer(n.timestamp))
            old_notifications=zip(old_notifications,time_since_creation_old)


    if new_notifications:
        for n in new_notifications:
            time_since_creation_new.append(time_differencer(n.timestamp))
        new_notifications=zip(new_notifications,time_since_creation_new)

    
    count= NotificationModelBuild.objects.count()
    
    # return {"notifications": notifications,"count":count, "time_since_creation": time_since_creation,"created_by":created_by,"messages":messages,"names":names,"type":type,"experiments":experiments}
    return {"new_notifications": new_notifications,"old_notifications":old_notifications,"count":count}




@register.simple_tag
def verbose_names(instance, field_name,name='verbose_names'):
    """
    Returns verbose_name for a field.
    """
    return instance._meta.get_field(field_name).verbose_name.title()