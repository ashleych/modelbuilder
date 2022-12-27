from django.test import TestCase
from django import setup

# Create your tests here.
# from django.test import TestCase
# import os

# os.environ.setdefault("DJANGO_SETTINGS_MODULE","model_builder.settings")
# setup()

import os

import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "model_builder.settings")

django.setup()
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