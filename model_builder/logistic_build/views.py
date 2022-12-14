from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.core.files.storage import default_storage
import os
import pandas as pd
from django.contrib.messages import constants as messages
#  Saving POST'ed file to storage
from .models import Traindata 
import json
# from .tasks import 
from django.http import (
    HttpResponse,
    HttpResponseGone,
    HttpResponseNotAllowed,
    HttpResponsePermanentRedirect,
    HttpResponseRedirect,
)
from django.template.response import TemplateResponse
from django.views import View
from django.views.generic.list import ListView
from django.views.generic.detail import DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.urls import reverse,reverse_lazy
from .models import Traindata,Variables,Experiment,Stationarity,Manualvariableselection,Classificationmodel
from .forms import ClassificationmodelForm, ExperimentForm,StationarityForm,ManualvariableselectionForm,ClassificationmodelForm


def index(request):
    # return render(request, 'logistic_build/layouts/base.html')
    # return render(request, 'logistic_build/index.html')
    return render(request, 'logistic_build/csv_inputs.html')

def upload_csv(request):
    data = {}
    # file = request.FILES['myfile']
    # file_name = default_storage.save(file.name, file)

    # #  Reading file from storage
    # file = default_storage.open(file_name)
    # file_url = default_storage.url(file_name)
    # file_data = csv_file.read().decode("utf-8")
    if "GET" == request.method:
        return render(request, "logistic_build/upload_csv.html", data)
    
    csv_file = request.FILES["portfolio_data_input"]
    macro_file = request.FILES["macro_file"]
    os.makedirs("input",exist_ok=True)
    portfolio_data_input= default_storage.save(os.path.join("input","input.csv"), csv_file)
    macro_file_name = default_storage.save(os.path.join("input","macro_input.csv"), macro_file)

    if not csv_file.name.endswith('.csv'):
        messages.error(request,'File is not CSV type')
        return HttpResponseRedirect(reverse("logistic_build"))
        

    colnames_macro =pd.read_csv(macro_file_name,nrows=20).columns.tolist()
    macro_file_obj=Traindata.objects.create(train_path = macro_file_name, train_data_name=request.POST.get("macro_input") )
    portfolio_file_obj=Traindata.objects.create(train_path = portfolio_data_input, train_data_name=request.POST.get("portfolio_data_input") )

    # e1=Experiment.objects.create(experiment_type='Input',name='macro_data_input',traindata=macro_file_obj)
    # relevant_col_names = TrainData.set_relevant_col_names(colnames_macro)
    for col in colnames_macro:
        pass
        # Variables.objects.create(var_name=col,file_id=macro_file_obj,experiment=e1,variable_type='Independent')
    return HttpResponse("Success !")



class TraindataBaseView(View):
    model = Traindata
    fields = '__all__'
    success_url = reverse_lazy('all')

class TraindataListView(TraindataBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Traindata objects"""

class TraindataDetailView(TraindataBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""

class TraindataCreateView(TraindataBaseView, CreateView):
    """View to create a new film"""

class TraindataUpdateView(TraindataBaseView, UpdateView):
    """View to update a film"""

class TraindataDeleteView(TraindataBaseView, DeleteView):
    """View to delete a film"""


def experiment_start(request):
    if "GET" == request.method:
        return render(request, "logistic_build/experiment_start.html")

class ExperimentFormView(CreateView):
    # model =Experiment
    form_class=ExperimentForm
    # fields= '__all__'
    template_name = 'logistic_build/stationarity_form.html'
    # def __init__(self, *args, **kwargs):
    #     super(ExperimentFormView, self).__init__(*args, **kwargs)
    # @method_decorator(login_required)
    # def dispatch(self, *args, **kwargs):
    #     return super(PlaceEventFormView, self).dispatch(*args, **kwargs)

    # def get_form_kwargs(self):
    #     kwargs = super(ExperimentFormView, self).get_form_kwargs()
        
    #     return kwargs

class ExperimentBaseView(View):
    Form = Experiment
    fields = '__all__'
    success_url = reverse_lazy('all')

class ExperimentListView(ExperimentBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Experiment objects"""
    model=Experiment
    
    def get_queryset(self):
        queryset =Experiment.objects.exclude(experiment_type='Input')
        return queryset

class ExperimentDetailView(ExperimentBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""

# class ExperimentCreateView(ExperimentBaseView, CreateView):
#     """View to create a new film"""

class ExperimentCreateView(ExperimentBaseView, CreateView):
    """View to create a new film"""
    # fields= ['name','traindata','do_kpss','do_adf','significance' ,'do_create_data']
class ExperimentUpdateView(ExperimentBaseView, UpdateView):
    """View to update a film"""

class ExperimentDeleteView(ExperimentBaseView, DeleteView):
    """View to delete a film"""
    success_url = reverse_lazy('experiment_all')

class StationarityBaseView(View):
    model = Stationarity
    fields = '__all__'
    success_url = reverse_lazy('all')

class StationarityListView(StationarityBaseView, ListView):
    
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Stationarity objects"""

class StationarityDetailView(StationarityBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""

class StationarityCreateView(StationarityBaseView, CreateView):
    """View to create a new film"""
    fields= [ "name", "traindata", "do_kpss", "do_adf", "significance" ,"do_create_data", "previous_experiment",]
class StationarityUpdateView( UpdateView):
    """View to update a film"""
    model=Stationarity
    form_class=StationarityForm

    def get_form_kwargs(self):
        kwargs = super(StationarityUpdateView, self).get_form_kwargs()
        return kwargs
class StationarityDeleteView(StationarityBaseView, DeleteView):
    """View to delete a film"""


class VariablesBaseView(View):
    model = Variables
    paginate_by = 10
    fields = '__all__'
    success_url = reverse_lazy('all')

class VariablesListView(VariablesBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Variables objects"""
    # Songs.objects.filter(genre__genre='Jazz')

    def get_queryset(self,**kwargs) :
        qs=super().get_queryset()
        return qs.filter(experiment__experiment_id=self.kwargs['experiment_id'])
        # p=qs.filter(experiment_id=5)
        # return qs.filter(experiment_id=7)


class VariablesDetailView(VariablesBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""

class VariablesCreateView(VariablesBaseView, CreateView):
    """View to create a new film"""

class VariablesUpdateView(VariablesBaseView, UpdateView):
    """View to update a film"""

class VariablesDeleteView(VariablesBaseView, DeleteView):
    """View to delete a film"""


class ManualvariableselectionBaseView(View):
    model = Manualvariableselection
    fields = '__all__'
    success_url = reverse_lazy('all')

class ManualvariableselectionListView(ManualvariableselectionBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Manualvariableselection objects"""

class ManualvariableselectionDetailView(ManualvariableselectionBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""

class ManualvariableselectionCreateView(ManualvariableselectionBaseView, CreateView):
    """View to create a new film"""

    fields= ['name','traindata']
class ManualvariableselectionUpdateView(UpdateView):
    """View to update a film"""
    model=Manualvariableselection
    form_class=ManualvariableselectionForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # context['load_template'] = 'assds'
        train_data_dict ={}
        for t in Traindata.objects.all().values():
            train_data_dict[t['file_id']]=t['column_names']
        context['train_data_dict']=json.dumps(train_data_dict)
        return context

    def get_form_kwargs(self):
        kwargs = super(ManualvariableselectionUpdateView, self).get_form_kwargs()
        return kwargs
class ManualvariableselectionDeleteView(ManualvariableselectionBaseView, DeleteView):
    """View to delete a film"""

class ManualvariableselectionFormView(CreateView):
    # model =Experiment
    form_class=ManualvariableselectionForm
    # fields= '__all__'
    template_name = 'logistic_build/stationarity_form.html'


class ClassificationmodelBaseView(View):
    model = Classificationmodel
    fields = '__all__'
    success_url = reverse_lazy('all')

class ClassificationmodelListView(ClassificationmodelBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Classificationmodel objects"""

class ClassificationmodelDetailView(ClassificationmodelBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""

class ClassificationmodelCreateView(ClassificationmodelBaseView, CreateView):
    """View to create a new film"""
    fields= ['name','traindata']

class ClassificationmodelUpdateView(UpdateView):
    """View to update a film"""
    model=Classificationmodel
    form_class=ClassificationmodelForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # context['load_template'] = 'assds'
        train_data_dict ={}
        for t in Traindata.objects.all().values():
            train_data_dict[t['file_id']]=t['column_names']
        context['train_data_dict']=json.dumps(train_data_dict)
        return context

    def get_form_kwargs(self):
        kwargs = super(ClassificationmodelUpdateView, self).get_form_kwargs()
        return kwargs
class ClassificationmodelDeleteView(ClassificationmodelBaseView, DeleteView):
    """View to delete a film"""

class ClassificationmodelFormView(CreateView):
    # model =Experiment
    form_class=ClassificationmodelForm
    # fields= '__all__'
    template_name = 'logistic_build/classificationmodel_form.html'



from django.http import JsonResponse
from time import sleep
from django.http import JsonResponse
from django_q.tasks import async_task

def index_j(request):
    json_payload = {
        "message": "Hello world!"
    }
    # sleep(10)
    async_task("logistic_build.tasks.sleep_and_print", 10)
    return JsonResponse(json_payload)