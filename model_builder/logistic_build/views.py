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
from .models import Traindata

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
    
    csv_file = request.FILES["csv_file"]
    macro_file = request.FILES["macro_file"]
    os.makedirs("input",exist_ok=True)
    
    csv_file_name = default_storage.save(os.path.join("input","input.csv"), csv_file)
    macro_file_name = default_storage.save(os.path.join("input","macro_input.csv"), macro_file)

    if not csv_file.name.endswith('.csv'):
        messages.error(request,'File is not CSV type')
        return HttpResponseRedirect(reverse("logistic_build"))
        

    # #if file is too large, return
    # if csv_file.multiple_chunks():
    #     messages.error(request,"Uploaded file is too big (%.2f MB)." % (csv_file.size/(1000*1000),))
    #     return HttpResponseRedirect(reverse("myapp:upload_csv"))
    # train_path = models.CharField(max_length=50)
    # colnames = models.CharField(max_length=50)
    # relevant_col_names 
    colnames_macro =pd.read_csv(macro_file_name,nrows=20).columns.tolist()
    Traindata.objects.create(train_path = macro_file_name, colnames = json.dumps(colnames_macro) )
    # relevant_col_names = TrainData.set_relevant_col_names(colnames_macro)

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