from django import forms

from django.forms import ModelForm
from .models import Experiment,Stationarity,Manualvariableselection,Classificationmodel,Regressionmodel,Featureselection


from django.forms import ModelForm, Textarea
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout,Row,Fieldset,Field,Div
from crispy_forms.bootstrap import FormActions

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Fieldset, Submit
from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

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
        m = super(ExperimentForm, self).save(commit=False, *args, **kwargs)

        models_dict = {'featureselection':Featureselection,'stationarity' : Stationarity, 'manualvariableselection' :Manualvariableselection,'classificationmodel':Classificationmodel,'regressionmodel':Regressionmodel}

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
import json
import ast
class ClassificationmodelForm(forms.ModelForm):
    experiment_type = forms.CharField(disabled=True,initial='Classification Model')
    # = forms.MultipleChoiceField()
    class Meta:
        model = Classificationmodel
        fields= [ "name", "traindata","testdata", "save_train_test_data", "experiment_type","previous_experiment", "run_in_the_background","label_col", "feature_cols", "train_split", "test_split", "feature_cols", "ignored_columns", "cross_validation","enable_spark","experiment_status"]
        widgets = {'experiment_status': forms.HiddenInput(),
        "feature_cols": forms.SelectMultiple(),
        "example_field": forms.SelectMultiple(choices= [('pi', 'PI'), ('ci', 'CI')])}
        # forms.CharField(required=False, widget=forms.SelectMultiple)

    def __init__(self, *args, **kwargs):
        # choices = kwargs.pop("choices")
        super().__init__(*args, **kwargs)
        if self.instance:
            # self.fields['feature_cols'].choices = json.loads(self.instance.feature_cols)
            if self.instance.feature_cols:
                choices= [(choice, choice) for choice in json.loads(self.instance.feature_cols)]
                self.fields['feature_cols'].choices = choices
                self.fields['feature_cols'].widget.choices = choices
        # self.request = kwargs.pop("request") # store value of request 
        self.fields['train_split'].widget.attrs['max'] = 1
        self.fields['train_split'].widget.attrs['min'] = 0
        # self.fields['example_field'].initial = ['pi']

        # print(self.request.user) 
    def clean_experiment_type(self):
        return 'classificationmodel'

    def clean(self):
        # form.cleaned_data['extra']
        cleaned_data = super().clean()
        cleaned_data['experiment_type']='classificationmodel'
        if not cleaned_data['testdata']:
            if not cleaned_data['train_split'] or not cleaned_data['test_split']:
                if not cleaned_data['testdata']:
                    raise forms.ValidationError("You need to enter train and test split or provide a test data set")
            if not float(cleaned_data['train_split']) + float(cleaned_data['test_split']) == 1:
                raise forms.ValidationError("Train and test splits need to add up to 1")
        
        cleaned_data['feature_cols'] = json.dumps(ast.literal_eval(cleaned_data['feature_cols']))
        # if self._errors and 'title' in self._errors:
        # cleaned_data['experiment_type']='newclassificationmodel'
        return cleaned_data

class RegressionmodelForm(forms.ModelForm):
   
    class Meta:
        model = Regressionmodel
        # fields = '__all__'
        fields= [ "name", "traindata" ,"do_create_data", "previous_experiment","run_in_the_background","label_col", "feature_cols", "train_split", "test_split", "feature_cols", "ignored_columns", "cross_validation","experiment_status"]
        widgets = {'experiment_status': forms.HiddenInput()}
    def clean(self):
        # form.cleaned_data['extra']
        cleaned_data = super().clean()
        return cleaned_data

class FeatureselectionForm(forms.ModelForm):
    class Meta:
        model = Featureselection
        fields = '__all__'
        # fields= [ "name", "traindata","do_create_data", "previous_experiment","run_in_the_background","label_col", "feature_cols", "train_split", "test_split", "feature_cols", "ignored_columns", "cross_validation","experiment_status"]
        # widgets = {'experiment_status': forms.HiddenInput()}
    
    def __init__(self, *args, **kwargs):
        super(FeatureselectionForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper(self)
        # self.helper.form_id = 'id-exampleForm'
        # self.helper.form_class = 'blueForms'
        # self.helper.form_method = 'post'
        self.helper.form_method = 'post'
        self.helper.form_action = 'Submit'
        self.helper.add_input(Submit('submit', 'Submit', css_class='btn btn-primary'))

        # self.helper.form_action = reverse('submit_form')
        # self.helper.form_action = 'submit_survey'
        # self.helper.render_required_fields=True

        self.helper.add_input(Submit('submit', 'Submit'))

        self.helper.layout = Layout(
            Fieldset("Objective","regression_or_classification"),
            Fieldset('Data Source','previous_experiment','traindata'),
            Fieldset('Select Column types','feature_cols','label_col'),
            Fieldset('Data Clean up','remove_constant_features','remove_quasi_constant_features','variance_threshold'),
            Fieldset('Feature Inclusions / Exclusions','fixed_features','ignored_features'),
            Fieldset("Correlation",'correlation_check','correlation_threshold'),
            Fieldset("Missing Value Treatment",'treat_missing' ),
            Fieldset('Variable Selection','max_features','min_features') ,
            Fieldset('Run Type','run_in_the_background','enable_spark','run_now') ,
        
            )
        
    def clean(self):
        # form.cleaned_data['extra']
        cleaned_data = super().clean()
        return cleaned_data
        
    def form_valid(self, form, *args, **kwargs):
        # user = form.cleaned_data.get('user')
        # self.log_id = User.objects.get(username=user).select_related('log').last().id
        return super().form_valid(form, *args, **kwargs) 

class ExampleForm(forms.Form):
    like_website = forms.TypedChoiceField(
        label = "Do you like this website?",
        choices = ((1, "Yes"), (0, "No")),
        coerce = lambda x: bool(int(x)),
        widget = forms.RadioSelect,
        initial = '1',
        required = True,
    )

    favorite_food = forms.CharField(
        label = "What is your favorite food?",
        max_length = 80,
        required = True,
    )

    favorite_color = forms.CharField(
        label = "What is your favorite color?",
        max_length = 80,
        required = True,
    )

    favorite_number = forms.IntegerField(
        label = "Favorite number",
        required = False,
    )

    notes = forms.CharField(
        label = "Additional notes or feedback",
        required = False,
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            Fieldset(
                'first arg is the legend of the fieldset',
                'like_website',
                'favorite_number',
                'favorite_color',
                'favorite_food',
                'notes'
            ),
            Submit('submit', 'Submit', css_class='button white'),
        )




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