from django.db import models
import json

from django.urls import reverse
# Create your models here.
from .stationarity_tests import adfuller_test,kpss_test
import pandas as pd
from django.core.files.storage import default_storage
import os
class Traindata(models.Model):
    file_id= models.AutoField(primary_key=True)
    train_data_name=models.CharField(max_length=100,null=True,blank=True) 
    train_path = models.CharField(max_length=50)
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
       return self.train_data_name

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




class Experiment(models.Model):
    EXPERIMENT_TYPE = [
        ('Input', 'Input'),
        ("Column_format_change",'Column_format_change'),
        ('Feature_engineering','Feature_engineering'),
        ('Stationarity_test', 'Stationarity_test'),

    ]
    experiment_id= models.AutoField(primary_key=True)
    experiment_type=models.CharField(max_length=100,choices=EXPERIMENT_TYPE,null=True, blank=True)
    name = models.CharField(max_length=100,null=True, blank=True)
    # start_date = models.DateField(null=True, blank=True)
    start_date= models.DateTimeField(auto_now_add=True, null=True, blank=True)
    
    traindata = models.ForeignKey(Traindata, on_delete=models.SET_NULL, null=True,related_name ='input_train_data')
    do_create_data =models.BooleanField(default=False)
    output_data = models.ForeignKey(Traindata, on_delete=models.SET_NULL, null=True,blank=True,related_name ='output_data')
    
    def __str__(self):
        return self.name
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
    traindata= models.ForeignKey(Traindata, on_delete=models.SET_NULL, null=True)
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
        return self.name
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


    def updates_variable_list():
        pass
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
            
    
    def create_variables_with_stationarity_results(self):
        for col in json.loads(self.stationarity_passed):
            Variables.objects.create(variable_name=col,traindata=self.traindata,experiment=self,experiment_status="PASS")

    def create_output_data(self):
        input_data=pd.read_csv(self.traindata.train_path)
        subset=input_data[json.loads(self.stationarity_passed)]
        os.makedirs("output",exist_ok=True)
        file_name_as_stored_on_disk= os.path.join("output",self.name+"_"+"Exp_id_"+ str(self.experiment_id) +".csv")
        if os.path.exists(file_name_as_stored_on_disk):
            os.remove(file_name_as_stored_on_disk)
        # portfolio_data_input= default_storage.save(os.path.join("input","input.csv"), csv_file)
        subset.to_csv(file_name_as_stored_on_disk)
        macro_file_obj=Traindata.objects.create(train_path = file_name_as_stored_on_disk, train_data_name=self.name + "_output") 
        self.output_data=macro_file_obj

    def save(self, *args, **kwargs):
            # self.slug = slugify(self.title)
            self.experiment_type='Stationarity_test'
            self.do_stationarity_test()
            super(Stationarity, self).save(*args, **kwargs)
            self.create_variables_with_stationarity_results()
            if self.do_create_data:
                self.create_output_data()
            super(Stationarity, self).save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('stationarity_detail', args=[str(self.experiment_id)])