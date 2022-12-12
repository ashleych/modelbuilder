from django.db import models
import json

from django.urls import reverse
# Create your models here.
from .stationarity_tests import adfuller_test,kpss_test
import pandas as pd
from django.core.files.storage import default_storage
import os
from django_q.tasks import async_task
from datetime import datetime
from django.conf import settings
from pathlib import Path
from django.utils import timezone

import datatable as dt
class Traindata(models.Model):
    file_id= models.AutoField(primary_key=True)
    train_data_name=models.CharField(max_length=100,null=True,blank=True) 
    train_path = models.CharField(max_length=50)
    no_of_cols=models.IntegerField(null=True,blank=True)
    no_of_rows=models.IntegerField(null=True,blank=True)
    column_names= models.CharField(max_length=50,null=True, blank=True)
    # colnames = models.CharField(max_length=54)
    # relevant_col_names = models.CharField(max_length=500)
    def get_absolute_url(self):
        return reverse('traindata_detail', args=[str(self.file_id)])
    # def set_relevant_col_names(self, x):
    #     self.relevant_col_names = json.dumps(x)

    # def get_relevant_col_names(self):
    #     return json.loads(self.relevant_col_names)

    # def get_col_names(self):
    #     return json.loads(self.colnames)
    def __str__(self):
        return self.train_data_name if self.train_data_name else ''

    def get_fields(self):
        return [(field.verbose_name, field.value_from_object(self)) for field in self.__class__._meta.fields[1:]] 
        # return [(field.verbose_name, field.value_from_object(self)) 
                
        #         if field.verbose_name != 'genre' 
        #             pass
        #         # else 
        #         #     (field.verbose_name, 
        #         #     Genre.objects.get(pk=field.value_from_object(self)).name)
                
        #         for field in self.__class__._meta.fields[1:]
        #     ] 

    def save(self, *args, **kwargs):
            # self.slug = slugify(self.title)
            # self.experiment_type='stationarity'
            t=dt.fread(file=self.train_path)
            self.no_of_cols=list(t.shape)[0]
            self.no_of_rows=list(t.shape)[1]
            self.column_names=json.dumps(list(t.names))
            super(Traindata, self).save(*args, **kwargs)


class Experiment(models.Model):
    EXPERIMENT_TYPE = [
        ('input', 'Input'),
        ("columnformatchange",'Column format change'),
        ('featureengineering','Feature engineering'),
        ('stationarity', 'Stationarity test'),
        ('manualvariableselection','Manual variable selection')

    ]
    STATUS_TYPE= [("STARTED","STARTED"),("IN_PROGRESS","IN PROGRESS"),("DONE","DONE"),("ERROR","ERROR")]
    experiment_id= models.AutoField(primary_key=True)
    experiment_type=models.CharField(max_length=100,choices=EXPERIMENT_TYPE,null=True, blank=True,default="input")
    name = models.CharField(max_length=100,null=True, blank=True)
    # start_date = models.DateField(null=True, blank=True)
    created_on_date= models.DateTimeField(auto_now_add=True, null=True, blank=True)
    experiment_status=models.CharField(max_length=20,choices=STATUS_TYPE,null=True, blank=True)
    traindata = models.ForeignKey(Traindata, on_delete=models.SET_NULL, null=True,blank=True, related_name ='input_train_data')
    do_create_data =models.BooleanField(default=False)
    output_data = models.ForeignKey(Traindata, on_delete=models.SET_NULL, null=True,blank=True,related_name ='output_data')
    previous_experiment = models.ForeignKey('self',null=True,blank=True,on_delete=models.SET_NULL)
    run_now=models.BooleanField(default=False)
    run_start_time= models.DateTimeField( null=True, blank=True)
    run_end_time= models.DateTimeField(null=True, blank=True)

   

    
    def __str__(self):
        return self.name if self.name else ''

    def get_fields(self):
        return [(field.verbose_name, field.value_from_object(self)) for field in self.__class__._meta.fields[1:]] 
        
    def get_absolute_url(self):
        return reverse('experiment_detail', args=[str(self.experiment_id)])
class Variables(models.Model):
    MODEL_VARIABLE_TYPE = [
        ('Dependent', 'Dependent'),
        ('Independent', 'Independent'),
    ]
    EXPERIMENT_STATUS_TYPE = [
        ('PASS', 'PASS'),
        ('FAIL', 'FAIL'),
    ]
    variable_id= models.AutoField(primary_key=True)
    variable_name=models.CharField(max_length=100,null=True, blank=True)
    traindata= models.ForeignKey(Traindata, on_delete=models.SET_NULL,blank=True, null=True)
    experiment= models.ForeignKey(Experiment, on_delete=models.SET_NULL, null=True)
    variable_type= models.CharField(
        max_length=20,
        choices=MODEL_VARIABLE_TYPE,
        default='Independent',
    )
    experiment_status= models.CharField(
        max_length=20,
        choices=EXPERIMENT_STATUS_TYPE,
        default='FAIL',
    )
    
    def __str__(self):
        return self.variable_name

    def get_fields(self):
        return [(field.verbose_name, field.value_from_object(self)) for field in self.__class__._meta.fields[1:]] 
    def get_absolute_url(self):
        return reverse('variables_detail', args=[str(self.variable_id)])


class Stationarity(Experiment):
    do_kpss=models.BooleanField(default=False)
    do_adf=models.BooleanField(default=False)
    significance= models.FloatField(null=True, blank=True,default=0.05)
    kpss_pass_vars=models.TextField(max_length=20000,blank=True,null=True)
    adf_pass_vars=models.TextField(max_length=20000,blank=True,null=True)
    stationarity_passed=models.TextField(max_length=20000,blank=True,null=True)

    def set_json(self, field,x):
        listVal= json.dumps(list(x))
        setattr(self,field,listVal)

    def get_json(self,field):
        return json.loads(getattr(self,field))


    def updates_variable_list(self):
        pass

            
    
    def create_variables_with_stationarity_results(self):
        if self.experiment_status:
            for col in json.loads(self.stationarity_passed):
                Variables.objects.create(variable_name=col,traindata=self.traindata,experiment=self,experiment_status="PASS")

    def create_output_data(self,*args,**kwargs):
        # import pdb; pdb.set_trace()
        # The path exists check is needed cos the background task do_stationary_tests calls this function as well whose cwd is different
        #ideally it should not be, but till the time the issue is fixed, one has to check if the path exists, else to go up one directory (whcih is where the tasks. py initiated workers have as directory)
        # essentially its whether the path is model_builder or model_builder/model_builder
        if os.path.exists(self.traindata.train_path):
            print( "path exists "+ self.traindata.train_path)
            file_path=self.traindata.train_path
            file_dir=""
        else:
            # print( "path exists "+ self.traindata.train_path)
            file_path=os.path.join(Path(settings.BASE_DIR).parent,self.traindata.train_path)
            file_dir=os.path.join(Path(settings.BASE_DIR).parent)
            print( "path exists "+ file_path)
            print(file_path)
        input_data=pd.read_csv(file_path)
        subset=input_data[json.loads(self.stationarity_passed)]
        out_dir_path=os.path.join(file_dir,"output")
        os.makedirs(out_dir_path,exist_ok=True)
        file_name_as_stored_on_disk= os.path.join(out_dir_path,self.name+"_"+"Exp_id_"+ str(self.experiment_id) +".csv")
        if os.path.exists(file_name_as_stored_on_disk):
            os.remove(file_name_as_stored_on_disk)
        print(file_name_as_stored_on_disk+" :    file_namesa as stpred ")
        # portfolio_data_input= default_storage.save(os.path.join("input","input.csv"), csv_file)
        subset.to_csv(os.path.join(file_name_as_stored_on_disk))
        macro_file_obj=Traindata.objects.create(train_path = file_name_as_stored_on_disk, train_data_name=self.name + "_output") 
        self.output_data=macro_file_obj
        # print("Create outputp data is being called ")

    def save(self, *args, **kwargs):
            # self.slug = slugify(self.title)
            self.experiment_type='stationarity'
            super(Stationarity, self).save(*args, **kwargs)
            if self.run_now:
                self.experiment_status='STARTED'
                super(Stationarity, self).save(*args, **kwargs)
                # from .tasks import do_stationarity_test_django_q
                # do_stationarity_test_django_q(self.experiment_id)
                async_task("logistic_build.tasks.do_stationarity_test_django_q", self.experiment_id)
                # async_task('tasks.create_html_report',
                #             request.user,
                #             hook='tasks.email_report')
            if self.experiment_status and self.experiment_status=='DONE':
                self.create_variables_with_stationarity_results()
                if self.do_create_data:
                    self.create_output_data()
                
                # self.experiment_status='DONE'
                self.run_end_time=timezone.now()
                super(Stationarity, self).save(*args, **kwargs)

    def get_absolute_url(self):
        if self.experiment_status:
            return reverse('stationarity_detail', args=[str(self.experiment_id)])
        else:
            return reverse('stationarity_update', args=[str(self.experiment_id)])

class Manualvariableselection(Experiment):

    input_columns=models.TextField(max_length=20000,blank=True,null=True)
    keep_columns=models.TextField(max_length=20000,blank=True,null=True)
    # def create_output_data(self):
    #     pass

    def subset_data(self):
        return 1

    def save(self, *args, **kwargs):
            # self.slug = slugify(self.title)
            if self._state.adding: 
                self.experiment_type='manualvariableselection'
                if self.previous_experiment and self.previous_experiment.output_data:
                    self.traindata=self.previous_experiment.output_data
                    self.input_columns=json.dumps(pd.read_csv(self.traindata.train_path,nrows=10).columns.to_list())

                    if  self.run_now:
                        self.run_start_time= timezone.now()
                        self.create_output_data()
                        self.run_end_time= timezone.now()
            else:
                if  self.run_now:
                        self.run_start_time= timezone.now()
                        self.create_output_data()
                        self.run_end_time= timezone.now()
                # df=self.subset_data()
            super(Manualvariableselection, self).save(*args, **kwargs)
            # if self.do_create_data:
            #         self.create_output_data(df)

    def get_absolute_url(self):
        if self.experiment_status:
            return reverse('manualvariableselection_detail', args=[str(self.experiment_id)])
        else:
            return reverse('manualvariableselection_update', args=[str(self.experiment_id)])

    def create_output_data(self):
        
        if self.run_now:
            if self.do_create_data:
                # self.create_output_data()
                cols_to_keep=json.loads(self.keep_columns)
                input_data= pd.read_csv(self.traindata.train_path)
                output_data = input_data[cols_to_keep]


                file_name_as_stored_on_disk= os.path.join("output",self.name+"_"+"Exp_id_"+ str(self.experiment_id) +".csv")
                if os.path.exists(file_name_as_stored_on_disk):
                    os.remove(file_name_as_stored_on_disk)
                output_data.to_csv(file_name_as_stored_on_disk)
                macro_file_obj=Traindata.objects.create(train_path = file_name_as_stored_on_disk, train_data_name=self.name + "_output")
                self.output_data=macro_file_obj
                self.experiment_status="DONE"

        # os.makedirs("output",exist_ok=True)

