from django import forms

from django.forms import ModelForm
from .models import Experiment,Stationarity,Manualvariableselection,Classificationmodel,Regressionmodel



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

        models_dict = {'stationarity' : Stationarity, 'manualvariableselection' :Manualvariableselection,'classificationmodel':Classificationmodel,'regressionmodel':Regressionmodel}
        s=models_dict[m.experiment_type].objects.create(experiment_type=m.experiment_type, name=m.name,previous_experiment=m.previous_experiment,created_by=m.created_by)
        if m.previous_experiment:
            s.traindata = m.previous_experiment.output_data
            s.save()
        # s.save()
        return s

class StationarityForm(forms.ModelForm):
   
    class Meta:
        model = Stationarity
        fields = '__all__'
        fields= [ "name", "traindata", "do_kpss", "do_adf", "significance" ,"do_create_data", "previous_experiment",'run_in_the_background']


    def clean(self):
        cleaned_data = super().clean()

class ManualvariableselectionForm(forms.ModelForm):
   
    class Meta:
        model = Manualvariableselection
        fields = '__all__'
        fields= [ "name", "traindata","keep_columns","do_create_data", "previous_experiment"]


    def clean(self):
        cleaned_data = super().clean()

class ClassificationmodelForm(forms.ModelForm):
   
    class Meta:
        model = Classificationmodel
        fields = '__all__'
        fields= [ "name", "traindata","do_create_data", "previous_experiment","run_in_the_background","label_col", "feature_cols", "train_split", "test_split", "feature_cols", "ignored_columns", "cross_validation","experiment_status"]
        widgets = {'experiment_status': forms.HiddenInput()}
    def clean(self):
        # form.cleaned_data['extra']
        cleaned_data = super().clean()
        return cleaned_data
class RegressionmodelForm(forms.ModelForm):
   
    class Meta:
        model = Regressionmodel
        fields = '__all__'
        fields= [ "name", "traindata","do_create_data", "previous_experiment","run_in_the_background","label_col", "feature_cols", "train_split", "test_split", "feature_cols", "ignored_columns", "cross_validation","experiment_status"]
        widgets = {'experiment_status': forms.HiddenInput()}
    def clean(self):
        # form.cleaned_data['extra']
        cleaned_data = super().clean()
        return cleaned_data
        

    # def form_valid(self, form, *args, **kwargs):
    #     # user = form.cleaned_data.get('user')
    #     # self.log_id = User.objects.get(username=user).select_related('log').last().id
    #     return super().form_valid(form, *args, **kwargs) 



from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Username",
                "class": "form-control"
            }
        ))
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password",
                "class": "form-control"
            }
        ))


class SignUpForm(UserCreationForm):
    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Username",
                "class": "form-control"
            }
        ))
    email = forms.EmailField(
        widget=forms.EmailInput(
            attrs={
                "placeholder": "Email",
                "class": "form-control"
            }
        ))
    password1 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password",
                "class": "form-control"
            }
        ))
    password2 = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                "placeholder": "Password check",
                "class": "form-control"
            }
        ))

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')