from django import forms

from django.forms import ModelForm
from .models import Experiment,Stationarity


# class ExperimentForm(forms.Form):
#     name = forms.CharField()
#     message = forms.CharField(widget=forms.Textarea)

#     def send_email(self):
#         # send email using the self.cleaned_data dictionary
#         pass

class ExperimentForm(ModelForm):
    class Meta:
        model = Experiment
        fields = ['name', 'experiment_type', 'previous_experiment']

    
    # def __init__(self, *args, **kwargs):
    #     # user = kwargs.pop('place_user')
    #     # now kwargs doesn't contain 'place_user', so we can safely pass it to the base class method
    #     super(ExperimentForm, self).__init__(*args, **kwargs)
        # self.fields['place'].queryset = Place.objects.filter(created_by=user)
    def save(self,*args,**kwargs):
        print("here")
        m = super(ExperimentForm, self).save(commit=False, *args, **kwargs)

        if m.experiment_type=='stationarity':
            s=Stationarity.objects.create(experiment_type=m.experiment_type, name=m.name,previous_experiment=m.previous_experiment)
            if m.previous_experiment:
                s.traindata = m.previous_experiment.traindata
                s.save()
            return s

class StationarityForm(forms.ModelForm):
    previous_experiment =  forms.CharField(disabled=True)
    class Meta:
        model = Stationarity
        fields = '__all__'
        fields= [ "name", "traindata", "do_kpss", "do_adf", "significance" ,"do_create_data", "previous_experiment",]