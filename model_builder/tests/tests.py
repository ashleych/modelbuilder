import pytest

from django.urls import reverse

import os
from django.conf import settings
from django.apps import apps

from logistic_build.models import Experiment,Classificationmodel,NotificationModelBuild,Traindata,Stationarity,Manualvariableselection, Regressionmodel
# https://code.visualstudio.com/docs/python/tutorial-django

from logistic_build.glr_spark import RegressionModel_spark,OverallRegressionResults,RegressionMetrics
@pytest.mark.django_db
def test_unauthorized(admin_client):
   url = reverse('traindata_list')
   response = admin_client.get(url)
   print(response)
   assert response.status_code == 200



def test_with_authenticated_client(client, django_user_model):
   username = "Siri"
   password = "siri"
   user = django_user_model.objects.create_user(username=username, password=password)
#    #  # Use this:
   # client.force_login(user)
    # Or this:
   client.login(username=username, password=password)
   url = reverse('traindata_list')
   response = client.get(url)
   # response = client.get('/private')
   assert response.status_code==200

from django.test import TestCase

from model_bakery import baker
from pprint import pprint

# class Experiment(TestCase):
#     def setUp(self):
#         self.customer = baker.make('shop.Customer')
#         pprint(self.customer.__dict__)


# pytest import
import pytest

# Third-party app imports
from model_bakery import baker

# from model_builder.models import Experiment

@pytest.fixture
def classificationmodel():
    """Fixture for baked Customer model."""
    pprint(baker.make(Classificationmodel))
    return baker.make(Classificationmodel)

@pytest.mark.django_db(transaction=True)
def test_using_customer( classificationmodel):
    """Test function using fixture of baked model."""
    assert isinstance( classificationmodel, Classificationmodel)

@pytest.fixture
def notificationmodelbuild():
    """Fixture for baked Customer model."""
    print(baker.make(NotificationModelBuild))
    return baker.make(NotificationModelBuild)

@pytest.mark.django_db(transaction=True)
def test_using_notifcation( notificationmodelbuild):
    """Test function using fixture of baked model."""
    assert isinstance( notificationmodelbuild, NotificationModelBuild)


# @pytest.mark.django_db(transaction=True)
# def test_using_customer( classificationmodel):
#    Traindata.obj

from logistic_build.models import Experiment,Classificationmodel,NotificationModelBuild,Traindata
# import pprint
class Classification(TestCase):
    def setUp(self):

        t=Traindata.objects.create(train_path='input/input_IRPqAaa.csv',train_data_name='new')
        Classificationmodel.objects.create(traindata=t,name='new',experiment_type='stationarity',label_col='def_trig',run_now=True,run_in_the_background=False)

    def test_classification_model_start(self):
        """Animals that can speak are correctly identified"""
        new_exp= Classificationmodel.objects.get(name="new")
        print(new_exp.__dict__)
        self.assertEqual(new_exp.run_now,True )

@pytest.mark.django_db()
def test_creating_customer( notificationmodelbuild):
    """Test function using fixture of baked model."""
    t=Traindata.objects.create(train_path='input/input_IRPqAaa.csv',train_data_name='new')
    t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    c=Classificationmodel.objects.create(traindata=t,name='new',experiment_type='stationarity',label_col='def_trig',run_now=True,run_in_the_background=False)
    c.run_now = True
    c.save()
    # assert isinstance( notificationmodelbuild, NotificationModelBuild)
    print(c)
    assert isinstance( c, Classificationmodel)


@pytest.mark.django_db()
def test_classifiction_model_view(client,django_user_model):
    
    """Test function using fixture of baked model."""
    t=Traindata.objects.create(train_path='input/input_IRPqAaa.csv',train_data_name='new')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    c=Classificationmodel.objects.create(traindata=t,name='new',experiment_type='classificationmodel',label_col='def_trig',run_now=True,run_in_the_background=False,created_by=user)
    c.run_now = True
    c.save()

    client.login(username=username, password=password)
    url = reverse('classificationmodel_detail', kwargs={'pk': c.experiment_id})
    response = client.get(url)
    # assert isinstance( notificationmodelbuild, NotificationModelBuild)
    assert(response.context_data['classificationmodel'].results_id >0)
    print(c)
    assert isinstance( c, Classificationmodel)


@pytest.mark.django_db()
def test_classifiction_preceding_experiment_view(client,django_user_model):
    
    """Test function using fixture of baked model."""
    t=Traindata.objects.create(train_path='input/input_IRPqAaa.csv',train_data_name='new')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    s=Stationarity.objects.create(traindata=t,name='new_stationary',experiment_type='stationarity',do_kpss=True,do_adf=True,run_now=False,run_in_the_background=False,created_by=user)
    f=Stationarity.objects.create(traindata=t,name='new_stat_2',experiment_type='stationarity',do_kpss=True,do_adf=True,run_now=False,run_in_the_background=False,created_by=user,previous_experiment=s)

    c=Classificationmodel.objects.create(traindata=t,name='new',experiment_type='classificationmodel',label_col='def_trig',run_now=True,run_in_the_background=False,created_by=user,previous_experiment=f)
    # c.run_now = True
    # c.save()

    client.login(username=username, password=password)
    url = reverse('classificationmodel_detail', kwargs={'pk': c.experiment_id})
    response = client.get(url)
    # assert isinstance( notificationmodelbuild, NotificationModelBuild)
    # assert(response.context_data['classificationmodel'].results_id >0)
    assert(response.context_data['classificationmodel'].all_preceding_experiments, "preceding experiments not populated")
    
    assert isinstance( c, Classificationmodel)




@pytest.mark.django_db()
def test_chain_muliple_experiments(client,django_user_model):
    
    """Test function using fixture of baked model."""
    t=Traindata.objects.create(train_path='input/input_IRPqAaa.csv',train_data_name='new')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    s=Stationarity.objects.create(traindata=t,name='new_stationary',experiment_type='stationarity',do_kpss=True,do_adf=True,run_now=False,run_in_the_background=False,created_by=user)
    f=Stationarity.objects.create(traindata=t,name='new_stat_2',experiment_type='stationarity',do_kpss=True,do_adf=True,run_now=False,run_in_the_background=False,created_by=user,previous_experiment=s)

    c=Classificationmodel.objects.create(traindata=t,name='new',experiment_type='classificationmodel',label_col='def_trig',run_now=True,run_in_the_background=False,created_by=user,previous_experiment=f)
    assert isinstance( c, Classificationmodel)
    c.run_now = True
    c.save()
    mvs = baker.make(Manualvariableselection,previous_experiment=c,created_by=user)

    client.login(username=username, password=password)
    url = reverse('classificationmodel_detail', kwargs={'pk': c.experiment_id})
    response = client.get(url)
    # assert isinstance( notificationmodelbuild, NotificationModelBuild)
    assert(response.context_data['classificationmodel'].all_preceding_experiments, "preceding experiments not populated")
    url = reverse('manualvariableselection_detail', kwargs={'pk': mvs.experiment_id})
    response = client.get(url)
    assert(response.context_data['manualvariableselection'].all_preceding_experiments, "preceding experiments not populated")
    # client.login(username=username, password=password)
    # assert(response.context_data['classificationmodel'].results_id >0)
    with open("yourhtmlfile_mvs.html", "wb") as file:
        file.write(response.content)



@pytest.fixture()
def classificationmodelbuild_save(django_user_model,django_db_blocker):
    """Fixture for baked Customer model."""
    t=Traindata.objects.create(train_path='input/input_IRPqAaa.csv',train_data_name='new')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri_2"
    password = "siri_2"
    user = django_user_model.objects.create_user(username=username, password=password)
    with django_db_blocker.unblock():
        c=Classificationmodel.objects.create(traindata=t,name='new',experiment_type='classificationmodel',label_col='def_trig',run_now=True,run_in_the_background=False,created_by=user)
        c.run_now = True
        c.save()
        pass
    return c

@pytest.mark.django_db(transaction=True)
def test_using_classificationmodelbuild( classificationmodelbuild_save):
    """Test function using fixture of baked model."""
    a=classificationmodelbuild_save
    assert isinstance( a, NotificationModelBuild)

import json
@pytest.mark.django_db()
def test_regression_spark_model(client,django_user_model):
    
    """Test function using fixture of baked model."""
    t=Traindata.objects.create(train_path='/home/ashleyubuntu/model_builder/car data.csv',train_data_name='new')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    s=Regressionmodel.objects.create(traindata=t,name='Regression Experiment',experiment_type='regression',
                                    label_col="Selling_Price",run_in_the_background=False,created_by=user,feature_cols=json.dumps(['Year', 'Present_Price', 'Kms_Driven', 'Owner']))
    s.label_col='Selling_Price'
    s.run_now=True
    s.save()
    exp_id=s.experiment_id
    saved_regression=Regressionmodel.objects.get(pk=exp_id)
    assert isinstance( s, Regressionmodel)

def test_regression_spark_function(client,django_user_model):
    
    """Test function using fixture of baked model."""
    file='/home/ashleyubuntu/model_builder/car data.csv'
    regression_result=RegressionModel_spark(filepath=file,label_col="Selling_Price",selected_columns=['Year', 'Present_Price', 'Kms_Driven', 'Owner'])
    assert isinstance(regression_result.overall_result, OverallRegressionResults)
    assert isinstance(regression_result.train_result, RegressionMetrics)
    assert isinstance(regression_result.test_result, RegressionMetrics)