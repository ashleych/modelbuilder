# from celery import Celery
from .stationarity_tests import adfuller_test,kpss_test
import pandas as pd
import json

from model_builder.celery import app
# from .models import Experiment,Stationarity,Logisticregression
import logistic_build.models as m
from pathlib import Path
import os
from django.utils import timezone

from .Logisticregression_spark import LogisticRegressionModel_spark,evaluation_metrics_mllib


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
        # import matplotlib
        # df_macros['M255'].plot.line()
        stationary_passed=set(kpss_results_passed).intersection(adf_results_passed)
        experiment.kpss_pass_vars=json.dumps(list(kpss_results_passed))
        experiment.adf_pass_vars=json.dumps(list(adf_results_passed))
        experiment.stationarity_passed=json.dumps(list(stationary_passed))
        experiment.experiment_status="DONE"
        if experiment.do_create_data:
            experiment.create_output_data(file_path=file_path)
        experiment.run_now=False
        experiment.save()
    
def run_logistic_regression(experiment_id,run_in_the_background=False):

        experiment = m.Classificationmodel.objects.get(experiment_id=experiment_id)
        print(settings.BASE_DIR)
        if os.path.exists(experiment.traindata.train_path):
            file_path=experiment.traindata.train_path
        else:
            file_path=os.path.join(Path(settings.BASE_DIR).parent,experiment.traindata.train_path)
            if not os.path.exists(file_path):
                ValueError("Input file doesnt exist")
        logistic_results = LogisticRegressionModel_spark(filepath=experiment.traindata.train_path, label_col=experiment.label_col ) 
        if run_in_the_background:
            print('inside the run in bck')
            train_results=m.ClassificationMetrics.objects.create(**logistic_results.train_result.all_attributes)
            test_results=m.ClassificationMetrics.objects.create(**logistic_results.test_result.all_attributes)

            experiment.results=m.ResultsClassificationmodel.objects.create(train_results=train_results,test_results=test_results,
                        coefficients=json.dumps(logistic_results.overall_result.coefficients),
                        feature_cols= json.dumps(logistic_results.overall_result.feature_cols))
            experiment.experiment_status='DONE'
            experiment.run_end_time= timezone.now()
            experiment.run_now=False
            experiment.save()
            # super(m.Classificationmodel, self).save(*args, **kwargs)
            experiment=m.Experiment.objects.get(experiment_id=experiment_id)
            notification=m.NotificationModelBuild.objects.create(is_read=False,timestamp=timezone.now(), message='Experiment Successful',experiment=experiment,created_by=experiment.created_by,experiment_type=experiment.experiment_type)
            return 
        return logistic_results
# from celery import shared_task

# MYGLOBAL = 0
# # @app.task(bind=True)
# @shared_task
# def fibonacci_task(experiment_id):
# # def fibonacci_task(self, experiment_id):
#     """Perform a calculation & update the status"""
#     # global MYGLOBAL
#     # MYGLOBAL=MYGLOBAL+1

#     calculation = Fibonnaci.objects.get(experiment_id=experiment_id)
#     print("calculation status from tasks",calculation.__dict__)
#     print("Self status from tasks",calculation.__dict__)

#     if not calculation.experiment_status=='DONE': 
#         try:
#             calculation.output = fib(calculation.number)
#             calculation.experiment_status = "DONE"
#         except Exception as e:
#             calculation.experiment_status = "ERROR"
#     # if not self.experiment_status=='DONE': 
#     #     try:
#     #         self.output = fib(calculation.input)
#     #         self.experiment_status = "DONE"
#     #     except Exception as e:
#     #         self.experiment_status = "ERROR"

#         calculation.save()        
#     # super(Fibonnaci,self).save({experiment_status:"DONE"})

