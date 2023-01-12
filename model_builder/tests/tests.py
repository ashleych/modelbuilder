from django import forms
import json
from django.test import TestCase
from pprint import pprint
from model_bakery import baker
import pytest

from django.urls import reverse

import os
from django.conf import settings
from django.apps import apps

from logistic_build.models import Experiment, Classificationmodel, NotificationModelBuild, Traindata, Stationarity, Manualvariableselection, Regressionmodel, Featureselection, TopModels

from logistic_build.forms import ClassificationmodelForm
# https://code.visualstudio.com/docs/python/tutorial-django

from logistic_build.glr_spark import RegressionModel_spark, OverallRegressionResults, RegressionMetrics
from logistic_build.scripts import sklearn_feature_selection
import pandas as pd

TESTING_DIRECTORY = '/home/ashleyubuntu/model_builder/data_for_tests'


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
    assert response.status_code == 200


# class Experiment(TestCase):
#     def setUp(self):
#         self.customer = baker.make('shop.Customer')
#         pprint(self.customer.__dict__)


# pytest import

# Third-party app imports

# from model_builder.models import Experiment


@pytest.fixture
def classificationmodel():
    """Fixture for baked Customer model."""
    pprint(baker.make(Classificationmodel))
    return baker.make(Classificationmodel)


@pytest.mark.django_db(transaction=True)
def test_using_customer(classificationmodel):
    """Test function using fixture of baked model."""
    assert isinstance(classificationmodel, Classificationmodel)


@pytest.fixture
def notificationmodelbuild():
    """Fixture for baked Customer model."""
    print(baker.make(NotificationModelBuild))
    return baker.make(NotificationModelBuild)


@pytest.mark.django_db(transaction=True)
def test_using_notifcation(notificationmodelbuild):
    """Test function using fixture of baked model."""
    assert isinstance(notificationmodelbuild, NotificationModelBuild)


# @pytest.mark.django_db(transaction=True)
# def test_using_customer( classificationmodel):
#    Traindata.obj

# import pprint


class Classification(TestCase):
    def setUp(self):

        t = Traindata.objects.create(train_path='input/input_IRPqAaa.csv', train_data_name='new')
        Classificationmodel.objects.create(traindata=t, name='new', experiment_type='stationarity', label_col='def_trig', run_now=True, run_in_the_background=False)

    def test_classification_model_start(self):
        """Animals that can speak are correctly identified"""
        new_exp = Classificationmodel.objects.get(name="new")
        print(new_exp.__dict__)
        self.assertEqual(new_exp.run_now, True)


@pytest.mark.django_db()
def test_creating_classification_model_with_feature_cols():
    """Test function using fixture of baked model."""
    t = Traindata.objects.create(train_path='input/input_IRPqAaa.csv', train_data_name='new')
    t = Traindata.objects.get(train_path='input/input_IRPqAaa.csv')

    #  testing with feature cols being sent in
    c = Classificationmodel.objects.create(traindata=t, name='new', feature_cols='["M97","M98"]', experiment_type='stationarity',
                                           label_col='def_trig', run_now=False, run_in_the_background=False, train_split=0.7, test_split=0.3)
    c.run_now = True
    c.save()
    assert isinstance(c, Classificationmodel)
    model_with_results = Classificationmodel.objects.get(pk=c.experiment_id)
    assert model_with_results.results_id, "model results not populated"

    # testing if notifications is created
    notification = NotificationModelBuild.objects.all()[0]
    assert isinstance(notification, NotificationModelBuild)


@pytest.mark.django_db()
def test_classifiction_model_view(client, django_user_model):
    """Test function using fixture of baked model."""
    t = Traindata.objects.create(train_path='input/input_IRPqAaa.csv', train_data_name='new')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    c = Classificationmodel.objects.create(traindata=t, name='new', experiment_type='classificationmodel', label_col='def_trig', run_now=True, run_in_the_background=False, created_by=user)
    c.run_now = True
    c.save()

    client.login(username=username, password=password)
    url = reverse('classificationmodel_detail', kwargs={'pk': c.experiment_id})
    response = client.get(url)
    # assert isinstance( notificationmodelbuild, NotificationModelBuild)
    assert (response.context_data['classificationmodel'].results_id > 0)
    print(c)
    assert isinstance(c, Classificationmodel)


@pytest.mark.django_db()
def test_classifiction_preceding_experiment_view(client, django_user_model):
    """Test function using fixture of baked model."""
    t = Traindata.objects.create(train_path='input/input_IRPqAaa.csv', train_data_name='new')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    s = Stationarity.objects.create(traindata=t, name='new_stationary', experiment_type='stationarity', do_kpss=True, do_adf=True, run_now=False, run_in_the_background=False, created_by=user)
    f = Stationarity.objects.create(traindata=t, name='new_stat_2', experiment_type='stationarity', do_kpss=True, do_adf=True,
                                    run_now=False, run_in_the_background=False, created_by=user, previous_experiment=s)

    c = Classificationmodel.objects.create(traindata=t, name='new', experiment_type='classificationmodel', label_col='def_trig',
                                           run_now=True, run_in_the_background=False, created_by=user, previous_experiment=f)

    client.login(username=username, password=password)
    url = reverse('classificationmodel_detail', kwargs={'pk': c.experiment_id})
    response = client.get(url)

    # assert(response.context_data['classificationmodel'].all_preceding_experiments is not None, "preceding experiments not populated")

    assert isinstance(c, Classificationmodel)


@pytest.mark.django_db()
def test_chain_muliple_experiments(client, django_user_model):
    """Test function using fixture of baked model."""
    t = Traindata.objects.create(train_path='input/input_IRPqAaa.csv', train_data_name='new')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    s = Stationarity.objects.create(traindata=t, name='new_stationary', experiment_type='stationarity', do_kpss=True, do_adf=True, run_now=False, run_in_the_background=False, created_by=user)
    f = Stationarity.objects.create(traindata=t, name='new_stat_2', experiment_type='stationarity', do_kpss=True, do_adf=True,
                                    run_now=False, run_in_the_background=False, created_by=user, previous_experiment=s)

    c = Classificationmodel.objects.create(traindata=t, name='new', experiment_type='classificationmodel', label_col='def_trig',
                                           run_now=True, run_in_the_background=False, created_by=user, previous_experiment=f)
    assert isinstance(c, Classificationmodel)
    c.run_now = True
    c.save()
    mvs = baker.make(Manualvariableselection, previous_experiment=c, created_by=user)

    client.login(username=username, password=password)
    url = reverse('classificationmodel_detail', kwargs={'pk': c.experiment_id})
    response = client.get(url)
    # assert isinstance( notificationmodelbuild, NotificationModelBuild)
    # assert (response.context_data['classificationmodel'].all_preceding_experiments is not None, "preceding experiments not populated")
    url = reverse('manualvariableselection_detail', kwargs={'pk': mvs.experiment_id})
    response = client.get(url)
    # assert (response.context_data['manualvariableselection'].all_preceding_experiments is not None, "preceding experiments not populated")
    # client.login(username=username, password=password)
    # assert(response.context_data['classificationmodel'].results_id >0)
    with open("yourhtmlfile_mvs.html", "wb") as file:
        file.write(response.content)


@pytest.fixture()
def classificationmodelbuild_save(django_user_model, django_db_blocker, enable_spark, train_data_path, label_col):
    """Fixture for baked Customer model."""
    t = Traindata.objects.create(train_path=train_data_path, train_data_name='new_train_data')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri_2"
    password = "siri_2"
    user = django_user_model.objects.create_user(username=username, password=password)
    # with django_db_blocker.unblock():
    c = Classificationmodel.objects.create(traindata=t, name='new', experiment_type='classificationmodel', label_col=label_col,
                                           run_now=True, run_in_the_background=False, created_by=user, enable_spark=enable_spark)
    c.run_now = True
    c.save()
    # pass
    return c


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize('enable_spark', [False])
@pytest.mark.parametrize('train_data_path', ['/home/ashleyubuntu/model_builder/sample_classification.csv'])
@pytest.mark.parametrize('label_col', ['TARGET'])
def test_using_classificationmodelbuild(client, classificationmodelbuild_save):
    """Test function using fixture of baked model."""
    a = classificationmodelbuild_save
    assert isinstance(a, Classificationmodel)


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize('enable_spark', [True])
@pytest.mark.parametrize('train_data_path', ['/home/ashleyubuntu/model_builder/sample_classification.csv'])
@pytest.mark.parametrize('label_col', ['TARGET'])
def test_using_classificationmodelbuild_with_spark(client, classificationmodelbuild_save):
    """Test function using fixture of baked model."""
    a = classificationmodelbuild_save
    assert isinstance(a, Classificationmodel)


@pytest.mark.django_db()
def test_regression_spark_model(client, django_user_model):
    """Test function using fixture of baked model."""
    t = Traindata.objects.create(train_path='/home/ashleyubuntu/model_builder/car data.csv', train_data_name='new')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    s = Regressionmodel.objects.create(traindata=t, name='Regression Experiment', experiment_type='regression',
                                       label_col="Selling_Price", run_in_the_background=False, created_by=user, feature_cols=json.dumps(['Year', 'Present_Price', 'Kms_Driven', 'Owner']))
    s.label_col = 'Selling_Price'
    s.run_now = True
    s.save()
    exp_id = s.experiment_id
    # have to read using exp_id from data base again as the save actually happens recuirsively.
    saved_regression = Regressionmodel.objects.get(pk=exp_id)
    assert isinstance(s, Regressionmodel)


def test_regression_spark_function(client, django_user_model):
    """Test function using fixture of baked model."""
    file = '/home/ashleyubuntu/model_builder/car data.csv'
    regression_result = RegressionModel_spark(filepath=file, label_col="Selling_Price", selected_columns=['Year', 'Present_Price', 'Kms_Driven', 'Owner'])
    assert isinstance(regression_result.overall_result, OverallRegressionResults)
    assert isinstance(regression_result.train_result, RegressionMetrics)
    assert isinstance(regression_result.test_result, RegressionMetrics)


@pytest.mark.django_db()
def test_multi_model_selection(client, django_user_model):
    fp = os.path.join(TESTING_DIRECTORY, 'santander_train_2k.csv')
    santander_data = Traindata.objects.create(train_path=fp, train_data_name='santander')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)

    s = Featureselection.objects.create(traindata=santander_data, name='Feat_selection', experiment_type='featuresselection',
                                        label_col="TARGET", run_in_the_background=False, created_by=user, cross_validation=0, max_features=3, short_list_max_features=3)
    s.run_now = True
    s.save()
    print(s)
    client.login(username=username, password=password)

    url = reverse('resultsfeatureselection_detail', kwargs={'pk': s.experiment_id})
    response = client.get(url)
    assert response.status_code == 200


@pytest.mark.django_db()
def test_multi_model_selection_2_factor_model(client, django_user_model):
    fp = os.path.join(TESTING_DIRECTORY, 'santander_train_2k.csv')
    santander_data = Traindata.objects.create(train_path=fp, train_data_name='santander')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)

    s = Featureselection.objects.create(traindata=santander_data, name='Feat_selection', experiment_type='featuresselection',
                                        label_col="TARGET", run_in_the_background=False, created_by=user, cross_validation=0, max_features=2, short_list_max_features=2)
    s.run_now = True
    s.save()
    print(s)
    client.login(username=username, password=password)

    url = reverse('resultsfeatureselection_detail', kwargs={'pk': s.experiment_id})
    response = client.get(url)
    assert response.status_code == 200


@pytest.mark.django_db()
def test_multi_model_selection_2_factor_model_with_model_build(client, django_user_model):
    fp = os.path.join(TESTING_DIRECTORY, 'santander_train_2k.csv')
    santander_data = Traindata.objects.create(train_path=fp, train_data_name='santander')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)

    s = Featureselection.objects.create(traindata=santander_data, name='Feat_selection', experiment_type='featuresselection',
                                        label_col="TARGET", run_in_the_background=False, created_by=user, cross_validation=0, max_features=2, short_list_max_features=2)
    s.run_now = True
    s.save()
    print(s)
    client.login(username=username, password=password)

    url = reverse('resultsfeatureselection_detail', kwargs={'pk': s.experiment_id})
    response = client.get(url)
    assert response.status_code == 200


# https://flowfx.de/blog/testing-django-forms-with-pytest-parameterization/


@pytest.mark.django_db()
def test_example_form(client, django_user_model):
    fp = os.path.join(TESTING_DIRECTORY, 'santander_train_2k.csv')
    santander_data = Traindata.objects.create(train_path=fp, train_data_name='santander')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    label_col = "TARGET"
    s = Featureselection.objects.create(traindata=santander_data, name='Feat_selection', experiment_type='featuresselection',
                                        label_col=label_col, run_in_the_background=False, created_by=user, cross_validation=0, max_features=2, short_list_max_features=2)
    s.run_now = True
    s.save()
    fs_saved = Featureselection.objects.get(pk=s.experiment_id)
    assert len(json.loads(fs_saved.results.shortlisted_features)) == 2, "Two variables werent short listed"

    topModel = TopModels.objects.filter(results=fs_saved.results).values()[0]
    topModel_id = topModel['id']

    client.login(username=username, password=password)

    url = reverse('classificationmodel_create', kwargs={"topmodel_id": topModel_id})
    response = client.get(url)
    assert response.context['form'].initial['label_col'] == label_col, 'Label col not populated in the form'
    post_data = response.context['form'].initial
    # form.is_valid()
    response.context['form'].is_valid()

    # post_data['_run_now'] = True  # to mimic when the user clicks the Submit button which is named _run_now, as opposed to the save as draft button
    response = client.post(url, data=data)

    assert response.status_code == 200
    # assert form.is_valid() is True


@pytest.mark.django_db()
def test_classification_model_form(client, django_user_model):
    fp = os.path.join(TESTING_DIRECTORY, 'santander_train_2k.csv')
    santander_data = Traindata.objects.create(train_path=fp, train_data_name='santander')

    data = {
        "name": 'bkpPaSKHtRwq6r3bnkM99e',
        "traindata": 1,
        "do_create_data": "on",
        "previous_experiment": '',
        "label_col": "TARGET",
        "feature_cols": json.dumps(["var3", "var15"]),
        "train_split": 0.7,
        "test_split": 0.3,
        "ignored_columns": '',
        "cross_validation": 5,
        "experiment_status": '',
        "_run_now": 'SUBMIT'}
    form = ClassificationmodelForm(data)
    # form.form_valid()
    # is_valid=form.is_valid()
    form.save()
    # assert is_valid


@pytest.mark.django_db()
def test_sample_post_classificationmodel(client, django_user_model):
    fp = os.path.join(TESTING_DIRECTORY, 'santander_train_2k.csv')
    santander_data = Traindata.objects.create(train_path=fp, train_data_name='santander')
    # t=Traindata.objects.get(train_path='input/input_IRPqAaa.csv')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    label_col = "TARGET"

    client.login(username=username, password=password)
    data = {
        "name": 'bkpPaSKHtRwq6r3bnkM99e',
        "traindata": 78,
        "do_create_data": "on",
        "previous_experiment": '',
        "label_col": "TARGET",
        "feature_cols": ["col_3", "col_11", "col_14"],
        "train_split": 0.7,
        "test_split": 0.3,
        "ignored_columns": '',
        "cross_validation": 5,
        "experiment_status": '',
        "_run_now": 'SUBMIT'}
    url = reverse('classificationmodel_create', kwargs={"topmodel_id": 1})
    # response = client.get(url)
    # assert response.context['form'].initial['label_col'] == label_col, 'Label col not populated in the form'
    post_data = data
    post_data.update({"traindata": 1})
    # response.context['form'].is_valid()
    # post_data['_run_now'] = True  # to mimic when the user clicks the Submit button which is named _run_now, as opposed to the save as draft button
    response = client.post(url, data=post_data)
    assert response.status_code == 200


@pytest.mark.parametrize('save_train_test_data', [True, False])
def test_sample_post_classificationmodel(client, django_user_model, save_train_test_data):
    fp = os.path.join(TESTING_DIRECTORY, 'santander_train_2k.csv')
    santander_data = Traindata.objects.create(train_path=fp, train_data_name='santander')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    label_col = "TARGET"
    feature_cols = ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1']

    c = Classificationmodel.objects.create(traindata=santander_data, name='new', feature_cols=json.dumps(feature_cols), experiment_type='classificationmodel',
                                           label_col=label_col, run_now=False, run_in_the_background=False, train_split=0.7, test_split=0.3, save_train_test_data=save_train_test_data)
    c.run_now = True
    c.save()
    assert isinstance(c, Classificationmodel)
    updated_experiment = Classificationmodel.objects.get(pk=c.experiment_id)
    assert isinstance(updated_experiment, Classificationmodel)
    assert updated_experiment.results, "results not populated"
    assert len(json.loads(json.loads(updated_experiment.results.coefficients))) == len(feature_cols)
    assert updated_experiment.results.train_nrows == 1400
    assert updated_experiment.results.test_nrows == 600
    assert updated_experiment.results.train_results.areaUnderROC > 0
    assert updated_experiment.results.train_results.precision is not None
    assert len(json.loads(updated_experiment.results.train_results.precision_plot_data)) > 0

import time
@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize('save_train_test_data', [True, False])
@pytest.mark.parametrize('run_in_the_background', [True, False])
def test_view_classificationmodel(client, django_user_model, save_train_test_data, run_in_the_background):
    fp = os.path.join(TESTING_DIRECTORY, 'santander_train_2k.csv')
    santander_data = Traindata.objects.create(train_path=fp, train_data_name='santander')
    username = "Siri"
    password = "siri"
    user = django_user_model.objects.create_user(username=username, password=password)
    label_col = "TARGET"
    feature_cols = ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1']
    print(run_in_the_background)
    c = Classificationmodel.objects.create(traindata=santander_data, name='new', feature_cols=json.dumps(feature_cols), experiment_type='classificationmodel', label_col=label_col, run_now=True, run_in_the_background = run_in_the_background, created_by=user, train_split=0.7, test_split=0.3, enable_spark=False, save_train_test_data=save_train_test_data)
    if run_in_the_background:
        assert c.results is None,"results not populated"
        time.sleep(10)
    assert isinstance(c, Classificationmodel)
    # assumption that the task would compete in 10 seconds
    refreshed_class_model= Classificationmodel.objects.get(pk=c.experiment_id)
    assert refreshed_class_model.results, "results still not done, try increasing the sleep time in the tests if needed"
    client.login(username=username, password=password)
    url = reverse('classificationmodel_detail', kwargs={"pk": c.experiment_id})
    response = client.get(url)
    assert response.status_code == 200
    with open("yourhtmlfile_mvs.html", "wb") as file:
        file.write(response.content)


@pytest.mark.django_db()
# @pytest.mark.parametrize('enable_spark', [False])
# @pytest.mark.parametrize('train_data_path', ['/home/ashleyubuntu/model_builder/sample_classification.csv'])
# @pytest.mark.parametrize('train_data_path', ['/home/ashleyubuntu/model_builder/sample_classification.csv'])
def test_model_classificationmodel(test_sample_post_classificationmodel):
    c = test_sample_post_classificationmodel()


# @pytest.fixture()
