from django import forms

from django.forms import ModelForm
from .models import Experiment,Stationarity,Manualvariableselection,Classificationmodel



class ExperimentForm(ModelForm):
    class Meta:
        model = Experiment
        fields = ['name', 'experiment_type','previous_experiment']
        # fields = ['name', 'experiment_type', 'previous_experiment']
    # def __init__(self, *args, **kwargs):
    #     # user = kwargs.pop('place_user')
    #     # now kwargs doesn't contain 'place_user', so we can safely pass it to the base class method
    #     super(ExperimentForm, self).__init__(*args, **kwargs)
        # self.fields['place'].queryset = Place.objects.filter(created_by=user)
    def save(self,*args,**kwargs):
        print("here")
        m = super(ExperimentForm, self).save(commit=False, *args, **kwargs)

        models_dict = {'stationarity' : Stationarity, 'manualvariableselection' :Manualvariableselection,'classificationmodel':Classificationmodel}
        s=models_dict[m.experiment_type].objects.create(experiment_type=m.experiment_type, name=m.name,previous_experiment=m.previous_experiment)
        if m.previous_experiment:
            s.traindata = m.previous_experiment.traindata
            s.save()
        # s.save()
        return s

class StationarityForm(forms.ModelForm):
   
    class Meta:
        model = Stationarity
        fields = '__all__'
        fields= [ "name", "traindata", "do_kpss", "do_adf", "significance" ,"do_create_data", "previous_experiment",'run_now']


    def clean(self):
        cleaned_data = super().clean()

class ManualvariableselectionForm(forms.ModelForm):
   
    class Meta:
        model = Manualvariableselection
        fields = '__all__'
        fields= [ "name", "traindata","keep_columns","do_create_data", "previous_experiment",'run_now']


    def clean(self):
        cleaned_data = super().clean()

class ClassificationmodelForm(forms.ModelForm):
   
    class Meta:
        model = Classificationmodel
        fields = '__all__'
        fields= [ "name", "traindata","do_create_data", "previous_experiment",'run_now',"label_col", "feature_cols", "train_split", "test_split", "feature_cols", "ignored_columns", "cross_validation"]

    def clean(self):
        cleaned_data = super().clean()