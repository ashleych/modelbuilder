from django import forms

from django.forms import ModelForm
from .models import Experiment,Stationarity,Manualvariableselection,Classificationmodel,Regressionmodel,Featureselection



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

class ClassificationmodelForm(forms.ModelForm):
   
    class Meta:
        model = Classificationmodel
        # fields = '__all__'
        fields= [ "name", "traindata","do_create_data", "previous_experiment","run_in_the_background","label_col", "feature_cols", "train_split", "test_split", "feature_cols", "ignored_columns", "cross_validation","enable_spark","experiment_status"]
        widgets = {'experiment_status': forms.HiddenInput()}
    def clean(self):
        # form.cleaned_data['extra']
        cleaned_data = super().clean()
        return cleaned_data
class RegressionmodelForm(forms.ModelForm):
   
    class Meta:
        model = Regressionmodel
        # fields = '__all__'
        fields= [ "name", "traindata","do_create_data", "previous_experiment","run_in_the_background","label_col", "feature_cols", "train_split", "test_split", "feature_cols", "ignored_columns", "cross_validation","experiment_status"]
        widgets = {'experiment_status': forms.HiddenInput()}
    def clean(self):
        # form.cleaned_data['extra']
        cleaned_data = super().clean()
        return cleaned_data

from django.forms import ModelForm, Textarea
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit, Layout,Row,Fieldset,Field,Div
from crispy_forms.bootstrap import FormActions
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
        


#    TASK_TYPE = [('regression','Regression'),('classification','Classification')]
#     label_col=models.CharField(max_length=100,blank=True,null=True)
#     feature_cols=models.TextField(max_length=20000,blank=True,null=True)
#     train_split=models.FloatField(blank=True,null=True)
#     test_split=models.FloatField(blank=True,null=True)
#     ignored_features=models.TextField(max_length=20000,blank=True,null=True)
#     fixed_features=models.TextField(max_length=20000,blank=True,null=True)
#     cross_validation=models.BooleanField(default=False)
#     results =models.ForeignKey(ResultsRegressionmodel, on_delete=models.CASCADE, null=True,blank=True)
#     short_list_max_features=models.FloatField(blank=True,null=True)
#     max_features=models.FloatField(blank=True,null=True)
#     min_features=models.FloatField(blank=True,null=True)
#     short_list_max_features=models.FloatField(blank=True,null=True)
#     regression_or_classification=models.CharField(max_length=20,choices=TASK_TYPE,default='regression')
#     remove_constant_features=models.BooleanField(default=False,null=True,blank=True)
#     remove_quasi_constant_features=models.BooleanField(default=False,null=True,blank=True)
#     variance_threshold=models.BooleanField(default=False,null=True,blank=True) #for quasi constant check
#     correlation_check =models.BooleanField(default=False,null=True,blank=True)
#     correlation_threshold=models.FloatField(blank=True,null=True)
#     treat_missing = models.BooleanField(default=False,null=True,blank=True)
#     variables_selected=  models.BooleanField(default=False,null=True,blank=True)
#     do_exhaustive_search = models.BooleanField(default=False,null=True,blank=True)
    def clean(self):
        # form.cleaned_data['extra']
        cleaned_data = super().clean()
        return cleaned_data
        

    def form_valid(self, form, *args, **kwargs):
        # user = form.cleaned_data.get('user')
        # self.log_id = User.objects.get(username=user).select_related('log').last().id
        return super().form_valid(form, *args, **kwargs) 
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Fieldset, Submit

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