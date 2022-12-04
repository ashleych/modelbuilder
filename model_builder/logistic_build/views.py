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

from django.views import View
from django.views.generic.list import ListView
from django.views.generic.detail import DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from .models import Traindata,Variables,Experiment,Stationarity

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

    e1=Experiment.objects.create(experiment_type='Input',name='macro_data_input',traindata=macro_file_obj)
    # relevant_col_names = TrainData.set_relevant_col_names(colnames_macro)
    for col in colnames_macro:
        Variables.objects.create(var_name=col,file_id=macro_file_obj,experiment=e1,variable_type='Independent')
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



class ExperimentBaseView(View):
    model = Experiment
    fields = '__all__'
    success_url = reverse_lazy('all')

class ExperimentListView(ExperimentBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Experiment objects"""

class ExperimentDetailView(ExperimentBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""

class ExperimentCreateView(ExperimentBaseView, CreateView):
    """View to create a new film"""

class ExperimentUpdateView(ExperimentBaseView, UpdateView):
    """View to update a film"""

class ExperimentDeleteView(ExperimentBaseView, DeleteView):
    """View to delete a film"""

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

    fields= ['name','traindata','do_kpss','do_adf','significance' ,'do_create_data']
class StationarityUpdateView(StationarityBaseView, UpdateView):
    """View to update a film"""

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