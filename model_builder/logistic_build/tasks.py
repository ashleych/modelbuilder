# from celery import Celery
from .stationarity_tests import adfuller_test,kpss_test
import pandas as pd
import json

# from model_builder.celery import app
import logistic_build.models as m
import os
from django.utils import timezone
from .Logisticregression_spark import LogisticRegressionModel_spark
from .Logisticregression_sklearn import LogisticRegressionModel_sklearn as lr_sk

from .glr_spark import RegressionModel_spark
from logistic_build.scripts import sklearn_multi_model_selection
from pathlib import Path
from django.conf import settings

def do_stationarity_test_django_q(experiment_id):
        experiment = m.Stationarity.objects.get(experiment_id=experiment_id)
        print(settings.BASE_DIR)
        if os.path.exists(experiment.traindata.train_path):
            file_path=experiment.traindata.train_path
            df_macros = pd.read_csv(experiment.traindata.train_path)
        else:
            file_path=os.path.join(Path(settings.BASE_DIR).parent,experiment.traindata.train_path)
            df_macros = pd.read_csv(file_path)

        numerics = ['int16', 'int32', 'int64','float64','float32']
        df = df_macros.select_dtypes(include=numerics)

        cols=df.columns.tolist()
        kpss_results=df_macros[cols].apply(lambda x: kpss_test(x), axis=0).T 
        kpss_results_passed=kpss_results[kpss_results['Accept (signif. level 0.05)']==True].index.values
        adf_results=df_macros[cols].apply(lambda x: adfuller_test(x), axis=0) 
        adf_results=adf_results.T
        adf_results_passed=adf_results[adf_results['Reject (signif. level 0.05)']==True].index.values
        
        stationary_passed=set(kpss_results_passed).intersection(adf_results_passed)
        experiment.kpss_pass_vars=json.dumps(list(kpss_results_passed))
        experiment.adf_pass_vars=json.dumps(list(adf_results_passed))
        experiment.stationarity_passed=json.dumps(list(stationary_passed))
        experiment.experiment_status="DONE"
        if experiment.do_create_data:
            experiment.create_output_data(file_path=file_path)
        experiment.run_now=False
        experiment.experiment_status='DONE'
        experiment.run_end_time= timezone.now()
        experiment.save()
        notification=m.NotificationModelBuild.objects.create(is_read=False,timestamp=timezone.now(), message='Stationarity Experiment Successful',experiment=experiment,created_by=experiment.created_by,experiment_type=experiment.experiment_type)
        return 
        
def run_logistic_regression(experiment_id):

        experiment = m.Classificationmodel.objects.get(experiment_id=experiment_id)
        print(settings.BASE_DIR)
        if os.path.exists(experiment.traindata.train_path):
            file_path=experiment.traindata.train_path
        else:
            file_path=os.path.join(Path(settings.BASE_DIR).parent,experiment.traindata.train_path)
            if not os.path.exists(file_path):
                ValueError("Input file doesnt exist")
        print(experiment.traindata.train_path)
        print(experiment.label_col)
        if experiment.enable_spark:
            logistic_results = LogisticRegressionModel_spark(filepath=experiment.traindata.train_path, label_col=experiment.label_col ) 
        else:
            logistic_results = lr_sk(filepath=experiment.traindata.train_path, label_col=experiment.label_col ) 
            
        if True:
            train_results=m.ClassificationMetrics.objects.create(**logistic_results.train_result.all_attributes)
            test_results=m.ClassificationMetrics.objects.create(**logistic_results.test_result.all_attributes)

            experiment.results=m.ResultsClassificationmodel.objects.create(train_results=train_results,test_results=test_results,
                        coefficients=json.dumps(logistic_results.overall_result.coefficients),
                        train_nrows=logistic_results.overall_result.train_nrows,
                        test_nrows=logistic_results.overall_result.test_nrows,
                        features= json.dumps(logistic_results.overall_result.features))
            experiment.experiment_status='DONE'
            experiment.run_end_time= timezone.now()
            experiment.run_now=False
            experiment.save()
            experiment=m.Experiment.objects.get(experiment_id=experiment_id)
            notification=m.NotificationModelBuild.objects.create(is_read=False,timestamp=timezone.now(), message='Experiment Successful',experiment=experiment,created_by=experiment.created_by,experiment_type=experiment.experiment_type)
            return 
        return logistic_results

def run_glm_regression(experiment_id):
        # This is run on Spark
        experiment = m.Regressionmodel.objects.get(experiment_id=experiment_id)
        
        if os.path.exists(experiment.traindata.train_path):
            file_path=experiment.traindata.train_path
        else:
            file_path=os.path.join(Path(settings.BASE_DIR).parent,experiment.traindata.train_path)
            if not os.path.exists(file_path):
                ValueError("Input file doesnt exist")
        if experiment.feature_cols:
            selected_cols= json.loads(experiment.feature_cols)
        else:
            selected_cols=None
        regression_results = RegressionModel_spark(filepath=experiment.traindata.train_path, label_col=experiment.label_col,selected_columns=selected_cols ) 
        if True:
            train_results=m.RegressionMetrics.objects.create(**regression_results.train_result.all_attributes)
            test_results=m.RegressionMetrics.objects.create(**regression_results.test_result.all_attributes)

            experiment.results=m.ResultsRegressionmodel.objects.create(train_results=train_results,test_results=test_results,**regression_results.overall_result.all_attributes)
            # experiment.results=m.ResultsRegressionmodel.objects.create(train_results=train_results,test_results=test_results,
            #             coefficients=json.dumps(regression_results.overall_result.coefficients),
            #             feature_cols= json.dumps(regression_results.overall_result.feature_cols))
            experiment.experiment_status='DONE'
            experiment.run_end_time= timezone.now()
            experiment.run_now=False
            experiment.save()
            experiment=m.Experiment.objects.get(experiment_id=experiment_id)
            notification=m.NotificationModelBuild.objects.create(is_read=False,timestamp=timezone.now(), message='Experiment Successful',experiment=experiment,created_by=experiment.created_by,experiment_type=experiment.experiment_type)
            return 
        return regression_results


def run_feature_selection(experiment_id):
        experiment = m.Featureselection.objects.get(experiment_id=experiment_id)
    
        feature_selection_results = sklearn_multi_model_selection.FeatureSelection(traindata_path=experiment.traindata.train_path,**vars(experiment) )
        model_subsets = feature_selection_results.models.subsets_
        experiment.experiment_status='DONE'
        experiment.run_end_time= timezone.now()
        experiment.run_now=False
        experiment.save()
        experiment=m.Experiment.objects.get(experiment_id=experiment_id)
        notification=m.NotificationModelBuild.objects.create(is_read=False,timestamp=timezone.now(), message='Feature selection Experiment Successful',experiment=experiment,created_by=experiment.created_by,experiment_type=experiment.experiment_type)
            # regression_results = sklearn_multi_model_selection.FeatureSelection(traindata=experiment.traindata.train_path, label_col=experiment.label_col,feature_cols=experiment.feature_cols,exclude_features=) 

        #     self.remove_constant_columns()
        # self.remove_quasi_constant_columns(variance_threshold=0.01)
        # # self.remove_duplicate_columns()
        # self.keep_numeric_columns_only()
        # self.remove_correlated_columns()
        # self.treat_nas()
        # self.random_forest_feature_selector()
        # self.exhaustive_model_builder()