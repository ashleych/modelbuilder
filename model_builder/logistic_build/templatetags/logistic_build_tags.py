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
    time_since_creation=[]
    number_of_notifications = 5
    # return NotificationModelBuild.objects.count()
    notifications=NotificationModelBuild.objects.all()[:number_of_notifications]
    time_since_creation=[]
    created_by =[]
    messages =[]
    names =[]
    experiments =[]
    type=[]
    for n in notifications:
       time_since_creation.append(time_differencer(n.timestamp))
       created_by.append(n.created_by)
       messages.append(n.message)
       names.append(n.experiment.name)
       experiments.append(n.experiment)
       type.append(n.experiment.experiment_type)
    count= NotificationModelBuild.objects.count()
    
    return {"notifications": notifications,"count":count, "time_since_creation": time_since_creation,"created_by":created_by,"messages":messages,"names":names,"type":type,"experiments":experiments}