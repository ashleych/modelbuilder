# from celery import Celery

# app = Celery('tasks', broker='pyamqp://guest@localhost//')

# @app.task
# def add(x, y):
#     return x + y
    
import random
from celery import shared_task

@shared_task
def add(x, y):
    # Celery recognizes this as the `movies.tasks.add` task
    # the name is purposefully omitted here.
    return x + y

@shared_task(name="multiply_two_numbers")
def mul(x, y):
    # Celery recognizes this as the `multiple_two_numbers` task
    total = x * (y * random.randint(3, 100))
    return total

@shared_task(name="sum_list_numbers")
def xsum(numbers):
    # Celery recognizes this as the `sum_list_numbers` task
    return sum(numbers)

from model_builder.celery import app
from .models import Experiment,Fibonnaci


def fib(n):
    """Calculate the Nth fibonacci number"""
    if n < 0:
        raise ValueError('Negative numbers are not supported')
    elif n == 0:
        return 0
    elif n <= 2:
        return 1

    return fib(n - 2) + fib(n - 1)


# from celery_tutorial.celery import app
# from .models import Calculation
from .stationarity_tests import adfuller_test,kpss_test
import pandas as pd
import json

# def fib(n):
#     """Calculate the Nth fibonacci number"""
#     if n < 0:
#         raise ValueError('Negative numbers are not supported')
#     elif n == 0:
#         return 0
#     elif n <= 2:
#         return 1

#     return fib(n - 2) + fib(n - 1)


# @app.task(bind=True)
# def fibonacci_task(self, calculation_id):
#     """Perform a calculation & update the status"""
#     calculation = Calculation.objects.get(id=calculation_id)

#     try:
#         calculation.output = fib(calculation.input)
#         calculation.status = Calculation.STATUS_SUCCESS
#     except Exception as e:
#         calculation.status = Calculation.STATUS_ERROR
#         calculation.message = str(e)[:110]

#     calculation.save()    


@app.task(bind=True)
def do_stationarity_test(self):
        df_macros = pd.read_csv(self.traindata.train_path)
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
        self.kpss_pass_vars=json.dumps(list(kpss_results_passed))
        self.adf_pass_vars=json.dumps(list(adf_results_passed))
        self.stationarity_passed=json.dumps(list(stationary_passed))
    

from celery import shared_task

MYGLOBAL = 0
# @app.task(bind=True)
@shared_task
def fibonacci_task(experiment_id):
# def fibonacci_task(self, experiment_id):
    """Perform a calculation & update the status"""
    # global MYGLOBAL
    # MYGLOBAL=MYGLOBAL+1

    calculation = Fibonnaci.objects.get(experiment_id=experiment_id)
    print("calculation status from tasks",calculation.__dict__)
    print("Self status from tasks",calculation.__dict__)
    print(MYGLOBAL)
    if not calculation.experiment_status=='DONE': 
        try:
            calculation.output = fib(calculation.number)
            calculation.experiment_status = "DONE"
        except Exception as e:
            calculation.experiment_status = "ERROR"
    # if not self.experiment_status=='DONE': 
    #     try:
    #         self.output = fib(calculation.input)
    #         self.experiment_status = "DONE"
    #     except Exception as e:
    #         self.experiment_status = "ERROR"

        calculation.save()        
    # super(Fibonnaci,self).save({experiment_status:"DONE"})