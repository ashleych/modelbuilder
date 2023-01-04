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
from django.views.generic import RedirectView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.urls import reverse,reverse_lazy
from .models import Traindata,Variables,Experiment,Stationarity,Manualvariableselection,Classificationmodel,ResultsClassificationmodel,NotificationModelBuild,RegressionMetrics,Regressionmodel,ResultsRegressionmodel,Featureselection
from .forms import ClassificationmodelForm, ExperimentForm, RegressionmodelForm,StationarityForm,ManualvariableselectionForm,ClassificationmodelForm,FeatureselectionForm
from .Logisticregression_spark import plot_roc,plot_precision_recall
from django.contrib.auth.mixins import LoginRequiredMixin
# from view_breadcrumbs import ListBreadcrumbMixin,DetailBreadcrumbMixin
from .forms import ExampleForm
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
    # macro_file = request.FILES["macro_file"]
    os.makedirs("input",exist_ok=True)
    portfolio_data_input= default_storage.save(os.path.join("input","input.csv"), csv_file)
    # macro_file_name = default_storage.save(os.path.join("input","macro_input.csv"), macro_file)

    if not csv_file.name.endswith('.csv'):
        messages.error(request,'File is not CSV type')
        return HttpResponseRedirect(reverse("logistic_build"))
        

    # colnames_macro =pd.read_csv(macro_file_name,nrows=20).columns.tolist()
    # macro_file_obj=Traindata.objects.create(train_path = macro_file_name, train_data_name=request.POST.get("macro_input") )
    portfolio_file_obj=Traindata.objects.create(train_path = portfolio_data_input, train_data_name=request.POST.get("portfolio_data_input") )

    # e1=Experiment.objects.create(experiment_type='Input',name='macro_data_input',traindata=macro_file_obj)
    # relevant_col_names = TrainData.set_relevant_col_names(colnames_macro)
    # for col in colnames_macro:
    #     pass
        # Variables.objects.create(var_name=col,file_id=macro_file_obj,experiment=e1,variable_type='Independent')
    return HttpResponse("Success !")

from django.http import JsonResponse
import json



class TraindataBaseView(LoginRequiredMixin,View):
    model = Traindata
    fields = '__all__'
    success_url = reverse_lazy('traindata_list')

from django.utils.functional import cached_property

class TraindataListView(TraindataBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Traindata objects"""
    template_name = 'logistic_build/traindata_list.html'

    @cached_property
    def crumbs(self):
        return [("My Test Breadcrumb", reverse("all"))]

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
    # form_class=ExampleForm
    # fields= '__all__'
    template_name = 'logistic_build/experiment_form.html'
    
    def form_valid(self, form):
        form.instance.created_by = self.request.user
        return super().form_valid(form)
    def __init__(self, *args, **kwargs):
        super(ExperimentFormView, self).__init__(*args, **kwargs)
    # @method_decorator(login_required)
    # def dispatch(self, *args, **kwargs):
    #     return super(PlaceEventFormView, self).dispatch(*args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super(ExperimentFormView, self).get_form_kwargs()
        
        return kwargs

class ExperimentBaseView(LoginRequiredMixin,View):
    Form = Experiment
    fields = '__all__'
    success_url = reverse_lazy('all')

class ExperimentListView(ExperimentBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Experiment objects"""
    model=Experiment
    paginate_by=10
    
    def get_queryset(self):
        queryset =Experiment.objects.exclude(experiment_type='Input')
        return queryset
    
    def get_context_data(self, **kwargs):
        s=super().get_context_data(**kwargs)
        return s

class ExperimentDetailView(ExperimentBaseView, RedirectView,DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""
    model=Experiment
    

    def get_redirect_url(self, *args, **kwargs):
        type=Experiment.objects.get(pk=kwargs['pk']).experiment_type
        return reverse(f'{type}_detail',kwargs={"pk": kwargs['pk']})
        # return reverse(f'{')
        user_role = Profile.objects.get(user=self.request.user).role
        if user_role in internal_users:
            return reverse('home')
        else:
            return reverse('event_list')
    # from django.http import Http404
    # https://stackoverflow.com/questions/6456586/redirect-from-generic-view-detailview-in-django
    # def get(self, request, *args, **kwargs):


class ExperimentCreateView(ExperimentBaseView, CreateView):
    """View to create a new film"""
    def get_context_data(self, **kwargs):
        context=super().get_context_data(**kwargs)
        return context
    # fields= ['name','traindata','do_kpss','do_adf','significance' ,'do_create_data']
class ExperimentUpdateView(ExperimentBaseView, UpdateView):
    """View to update a film"""

class ExperimentDeleteView(ExperimentBaseView, DeleteView):
    """View to delete a film"""
    success_url = reverse_lazy('experiment_all')
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # context['load_template'] = 'assds'
        # train_data_dict ={}
        # for t in Traindata.objects.all().values():
        #     train_data_dict[t['file_id']]=t['column_names']
        # context['train_data_dict']=json.dumps(train_data_dict)
        return context

    # def get_queryset(self):
    #     # queryset =Experiment.objects.exclude(experiment_type='Input')
    #     queryset=Experiment.objects.get(pk=self.__dict__['kwargs']['pk'])
    #     return queryset
class StationarityBaseView(LoginRequiredMixin,View):
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
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['created_by']=context['stationarity'].created_by.username

        if context['stationarity'].all_preceding_experiments:
            context['previous_experiments_list']=json.loads(context['stationarity'].all_preceding_experiments)
            context['current_experiment_list']=tuple([context['stationarity'].experiment_id,context['stationarity'].experiment_type,context['stationarity'].name])

        return context
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

    def form_valid(self, form):
        if '_run_now' in form.data:
            form.instance.run_now=True   #     this is done so that,when user clicks 'save as draft' then run_now is kept as false
        form.instance.experiment_status='NOT_STARTED'
        return super().form_valid(form)
class StationarityDeleteView(StationarityBaseView, DeleteView):
    """View to delete a film"""


class VariablesBaseView(View):
    model = Variables
    paginate_by = 10
    fields = '__all__'
    success_url = reverse_lazy('all')

class VariablesListView(LoginRequiredMixin,VariablesBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Variables objects"""
    # Songs.objects.filter(genre__genre='Jazz')

    def get_queryset(self,**kwargs) :
        qs=super().get_queryset()
        return qs.filter(experiment__experiment_id=self.kwargs['experiment_id'])
        # p=qs.filter(experiment_id=5)
        # return qs.filter(experiment_id=7)


class VariablesDetailView(LoginRequiredMixin,VariablesBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""

class VariablesCreateView(LoginRequiredMixin,VariablesBaseView, CreateView):
    """View to create a new film"""

class VariablesUpdateView(LoginRequiredMixin,VariablesBaseView, UpdateView):
    """View to update a film"""

class VariablesDeleteView(LoginRequiredMixin,VariablesBaseView, DeleteView):
    """View to delete a film"""


class ManualvariableselectionBaseView(View):
    model = Manualvariableselection
    fields = '__all__'
    success_url = reverse_lazy('all')

class ManualvariableselectionListView(LoginRequiredMixin,ManualvariableselectionBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Manualvariableselection objects"""

class ManualvariableselectionDetailView(LoginRequiredMixin,ManualvariableselectionBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        if context['manualvariableselection'].created_by:
            context['created_by']=context['manualvariableselection'].created_by.username
        type='manualvariableselection'
        if context[type].all_preceding_experiments:
            context['previous_experiments_list']=json.loads(context[type].all_preceding_experiments)
            context['current_experiment_list']=tuple([context[type].experiment_id,context[type].experiment_type,context[type].name])

        return context
    

class ManualvariableselectionCreateView(LoginRequiredMixin,ManualvariableselectionBaseView, CreateView):
    """View to create a new film"""

    fields= ['name','traindata']
class ManualvariableselectionUpdateView(LoginRequiredMixin,UpdateView):
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
    
    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        # form.send_email()
        if '_run_now' in form.data:
            form.instance.run_now=True   #     this is done so that,when user clicks 'save as draft' then run_now is kept as false
        form.instance.experiment_status='NOT_STARTED'
        return super().form_valid(form)
class ManualvariableselectionDeleteView(LoginRequiredMixin,ManualvariableselectionBaseView, DeleteView):
    """View to delete a film"""

class ManualvariableselectionFormView(LoginRequiredMixin,CreateView):
    # model =Experiment
    form_class=ManualvariableselectionForm
    # fields= '__all__'
    template_name = 'logistic_build/stationarity_form.html'


class ClassificationmodelBaseView(View):
    model = Classificationmodel
    fields = '__all__'
    success_url = reverse_lazy('all')

class ClassificationmodelListView(LoginRequiredMixin,ClassificationmodelBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Classificationmodel objects"""

class ClassificationmodelDetailView(LoginRequiredMixin,ClassificationmodelBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['created_by']=context['classificationmodel'].created_by.username
        if context['classificationmodel'].all_preceding_experiments:
            context['previous_experiments_list']=json.loads(context['classificationmodel'].all_preceding_experiments)
            context['current_experiment_list']=tuple([context['classificationmodel'].experiment_id,context['classificationmodel'].experiment_type,context['classificationmodel'].name])

        return context
class ClassificationmodelCreateView(LoginRequiredMixin,ClassificationmodelBaseView, CreateView):
    """View to create a new film"""
    fields= ['name','traindata']

class ClassificationmodelUpdateView(LoginRequiredMixin,UpdateView):
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
        # if 'data' in kwargs: #or check if self.request.method = POST
        #     if '_run_now' in kwargs['data']:
        #         kwargs['run_now']=['on']
        return kwargs
    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        # form.send_email()
        if '_run_now' in form.data:
            form.instance.run_now=True   #     this is done so that,when user clicks 'save as draft' then run_now is kept as false
        form.instance.experiment_status='NOT_STARTED'
        return super().form_valid(form)
    
    # def get_success_url(self):
    #     a=1
    #     return a

class ClassificationmodelDeleteView(LoginRequiredMixin,ClassificationmodelBaseView, DeleteView):
    """View to delete a film"""
    template_name='logistic_build/experiment_confirm_delete.html'
    
    def get_context_data(self, **kwargs) :
        context=super().get_context_data(**kwargs)
        context['name']=context['object'].name
        return context


class ClassificationmodelFormView(LoginRequiredMixin,CreateView):
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

class ResultsClassificationmodelBaseView(View):
    model = ResultsClassificationmodel
    fields = '__all__'
    success_url = reverse_lazy('all')


class ResultsClassificationmodelDetailView(LoginRequiredMixin,ResultsClassificationmodelBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""

    def get_context_data(self, **kwargs) :
        from plotly.offline import plot
        context=super().get_context_data(**kwargs)

        def get_fpr_tpr(result, type ='train'):
            view_dict={}
            fpr=json.loads(getattr(getattr(result,type+'_results'),'FPR'))
            tpr=json.loads(getattr(getattr(result,type+'_results'),'TPR'))
            auc=getattr(getattr(result,type+'_results'),'areaUnderROC')
            fig=plot_roc(fpr,tpr,auc)
            plotly_plot_obj = plot({'data': fig}, output_type='div')
            view_dict['rocPlot']=plotly_plot_obj
            view_dict['auc']=auc
            return view_dict

        def get_precision_recall_plot(result, type ='train'):
            view_dict={}
            precision=json.loads(getattr(getattr(result,type+'_results'),'precision_plot_data'))
            recall=json.loads(getattr(getattr(result,type+'_results'),'recall_plot_data'))
            pr=getattr(getattr(result,type+'_results'),'areaUnderPR')
            fig=plot_precision_recall(recall_plot_data=recall,precision_plot_data=precision,areaUnderPR=pr)

            plotly_plot_obj = plot({'data': fig}, output_type='div')
            view_dict['prPlot']=plotly_plot_obj
            view_dict['pr']=pr
            return view_dict
        res =context['resultsclassificationmodel']
        context['coefficients']=context['resultsclassificationmodel'].coefficients
        context['features']=context['resultsclassificationmodel'].features
        context['train_nrows']=context['resultsclassificationmodel'].train_nrows
        context['test_nrows']=context['resultsclassificationmodel'].test_nrows


        context['train_res_roc']=get_fpr_tpr(res,type='train')

        context['train_res_pr']=get_precision_recall_plot(res,type='train')
        if res.test_results.areaUnderROC:
            context['test_res_roc']=get_fpr_tpr(res,type='test')
            context['test_res_pr']=get_precision_recall_plot(res,type='test')
        else:
            context['test_res_roc']=None
            context['test_res_pr']=None
        
        unread_notifications = NotificationModelBuild.objects.filter(is_read=False).count()
        context["unread_notifications"] = unread_notifications
        return context


from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.forms.utils import ErrorList
from django.http import HttpResponse
from .forms import LoginForm, SignUpForm

def login_view(request):
    form = LoginForm(request.POST or None)

    msg = None

    if request.method == "POST":

        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect("/")
            else:    
                msg = 'Invalid credentials'    
        else:
            msg = 'Error validating the form'    

    return render(request, "accounts/login.html", {"form": form, "msg" : msg})

def register_user(request):

    msg     = None
    success = False

    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get("username")
            raw_password = form.cleaned_data.get("password1")
            user = authenticate(username=username, password=raw_password)

            msg     = 'User created - please <a href="/login">login</a>.'
            success = True
            
            #return redirect("/login/")

        else:
            msg = 'Form is not valid'    
    else:
        form = SignUpForm()

    return render(request, "accounts/register.html", {"form": form, "msg" : msg, "success" : success })



class RegressionmodelBaseView(View):
    model = Regressionmodel
    fields = '__all__'
    success_url = reverse_lazy('all')

class RegressionmodelListView(LoginRequiredMixin,RegressionmodelBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Regressionmodel objects"""

class RegressionmodelDetailView(LoginRequiredMixin,RegressionmodelBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['created_by']=context['regressionmodel'].created_by.username
        if context['regressionmodel'].all_preceding_experiments:
            context['previous_experiments_list']=json.loads(context['regressionmodel'].all_preceding_experiments)
            context['current_experiment_list']=tuple([context['regressionmodel'].experiment_id,context['regressionmodel'].experiment_type,context['classificationmodel'].name])

        return context
class RegressionmodelCreateView(LoginRequiredMixin,RegressionmodelBaseView, CreateView):
    """View to create a new film"""
    fields= ['name','traindata']

class RegressionmodelUpdateView(LoginRequiredMixin,UpdateView):
    """View to update a film"""
    model=Regressionmodel
    form_class=RegressionmodelForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # context['load_template'] = 'assds'
        train_data_dict ={}
        for t in Traindata.objects.all().values():
            train_data_dict[t['file_id']]=t['column_names']
        context['train_data_dict']=json.dumps(train_data_dict)
        return context

    def get_form_kwargs(self):
        kwargs = super(RegressionmodelUpdateView, self).get_form_kwargs()
        # if 'data' in kwargs: #or check if self.request.method = POST
        #     if '_run_now' in kwargs['data']:
        #         kwargs['run_now']=['on']
        return kwargs
    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        # form.send_email()
        if '_run_now' in form.data:
            form.instance.run_now=True   #     this is done so that,when user clicks 'save as draft' then run_now is kept as false
        form.instance.experiment_status='NOT_STARTED'
        return super().form_valid(form)
    
    # def get_success_url(self):
    #     a=1
    #     return a

class RegressionmodelDeleteView(LoginRequiredMixin,RegressionmodelBaseView, DeleteView):
    """View to delete a film"""
    template_name='logistic_build/experiment_confirm_delete.html'
    
    def get_context_data(self, **kwargs) :
        context=super().get_context_data(**kwargs)
        context['name']=context['object'].name
        return context


class RegressionmodelFormView(LoginRequiredMixin,CreateView):
    # model =Experiment
    form_class=RegressionmodelForm
    # fields= '__all__'
    template_name = 'logistic_build/regressionmodel_form.html'


class FeatureselectionBaseView(View):
    model = Featureselection
    fields = '__all__'
    success_url = reverse_lazy('all')

class FeatureselectionListView(LoginRequiredMixin,FeatureselectionBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all Featureselection objects"""

class FeatureselectionDetailView(LoginRequiredMixin,FeatureselectionBaseView, DetailView):
    """View to list the details from one film.
    Use the 'film' variable in the template to access
    the specific film here and in the Views below"""
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['created_by']=context['featureselection'].created_by.username
        if context['featureselection'].all_preceding_experiments:
            context['previous_experiments_list']=json.loads(context['featureselection'].all_preceding_experiments)
            context['current_experiment_list']=tuple([context['featureselection'].experiment_id,context['featureselection'].experiment_type,context['classificationmodel'].name])

        return context
class FeatureselectionCreateView(LoginRequiredMixin,FeatureselectionBaseView, CreateView):
    """View to create a new film"""
    fields= ['name','traindata']

class FeatureselectionUpdateView(LoginRequiredMixin,UpdateView):
    """View to update a film"""
    model=Featureselection 
    # form_class=FeatureselectionForm
    form_class=FeatureselectionForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # context['load_template'] = 'assds'
        train_data_dict ={}
        for t in Traindata.objects.all().values():
            train_data_dict[t['file_id']]=t['column_names']
        context['train_data_dict']=json.dumps(train_data_dict)
        return context

    def get_form_kwargs(self):
        kwargs = super(FeatureselectionUpdateView, self).get_form_kwargs()
        # if 'data' in kwargs: #or check if self.request.method = POST
        #     if '_run_now' in kwargs['data']:
        #         kwargs['run_now']=['on']
        return kwargs
    def form_valid(self, form):
        # This method is called when valid form data has been POSTed.
        # It should return an HttpResponse.
        # form.send_email()
        if '_run_now' in form.data:
            form.instance.run_now=True   #     this is done so that,when user clicks 'save as draft' then run_now is kept as false
        form.instance.experiment_status='NOT_STARTED'
        return super().form_valid(form)
    
    # def get_success_url(self):
    #     a=1
    #     return a

class FeatureselectionDeleteView(LoginRequiredMixin,FeatureselectionBaseView, DeleteView):
    """View to delete a film"""
    template_name='logistic_build/experiment_confirm_delete.html'
    
    def get_context_data(self, **kwargs) :
        context=super().get_context_data(**kwargs)
        context['name']=context['object'].name
        return context


class FeatureselectionFormView(LoginRequiredMixin,CreateView):
    # model =Experiment
    form_class=FeatureselectionForm
    # fields= '__all__'
    template_name = 'logistic_build/featureselection_form.html'

class NotificationModelBuildBaseView(View):
    model = NotificationModelBuild
    paginate_by = 10
    fields = '__all__'
    success_url = reverse_lazy('all')

class NotificationModelBuildListView(LoginRequiredMixin,NotificationModelBuildBaseView, ListView):
    """View to list all films.
    Use the 'film_list' variable in the template
    to access all NotificationModelBuild objects"""
    # Songs.objects.filter(genre__genre='Jazz')

class NotificationModelBuildUpdateView(LoginRequiredMixin,NotificationModelBuildBaseView, UpdateView):
    """View to update a film"""

class NotificationModelBuildDeleteView(LoginRequiredMixin,NotificationModelBuildBaseView, DeleteView):
    """View to delete a film"""




def notifications_mark_as_read(request):


        notifications = NotificationModelBuild.objects.filter(created_by=request.user)
        if notifications:
            notifications.update(is_read=True)  
        # return JsonResponse(data)
        return HttpResponse("Success !")

def notifications_delete(request):

        notifications = NotificationModelBuild.objects.filter(created_by=request.user).delete()

        # return JsonResponse(data)
        return HttpResponse("Success !")
