import pytest

from django.urls import reverse

import os
from django.conf import settings
from django.apps import apps

from logistic_build.models import Experiment,Classificationmodel,NotificationModelBuild,Traindata



@pytest.mark.django_db
def test_unauthorized(admin_client):
   url = reverse('traindata_list')
   response = admin_client.get(url)
   print(response)
   assert response.status_code == 200



def test_with_authenticated_client(client, django_user_model):
   username = "Siri"
   password = "siri"
   # user = django_user_model.objects.create_user(username=username, password=password)
   #  # Use this:
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
    # assert isinstance( notificationmodelbuild, NotificationModelBuild)
    print(c)
    assert isinstance( c, Classificationmodel)