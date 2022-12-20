from django.contrib import admin

# Register your models here.

from .models import Traindata,Variables,Experiment,Stationarity,Manualvariableselection,Classificationmodel,ResultsClassificationmodel,Notification


admin.site.register(Traindata)
admin.site.register(Variables)
admin.site.register(Experiment)
admin.site.register(Stationarity)
admin.site.register(Manualvariableselection)
admin.site.register(Classificationmodel)
admin.site.register(ResultsClassificationmodel)
admin.site.register(Notification)


# class ExperimentAdmin(admin.ModelAdmin):
#     # list_display = ('title', 'author')
#     # fieldsets = [
#     #     (None, { 'fields': [('title','body')] } ),
#     # ]

#     def save_model(self, request, obj, form, change):
#         if getattr(obj, 'author', None) is None:
#             obj.author = request.user
#         obj.save()