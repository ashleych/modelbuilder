from django.db import models
import json

from django.urls import reverse

from .utils import get_spark_session
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
from django.contrib.auth.models import User
import os
import glob
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

            # if file details are already available, there is no need to re-read these details
            # file details are available in case of spark operations, where output file is written by spark
            # in case the file is directly input via the front end, this infor will be missing and hence will have to be read in the save here
            if not self.no_of_cols:
                if os.path.isdir(self.train_path):
                    # Path can be a directory in case it has multiple files. Usually this willbe because these are csv files from a spark write operation
                    # for now it is assumed that these are all csv files and spark will be used to read these files
                    pass
                else:
                    t=dt.fread(file=self.train_path)
                    self.no_of_cols=list(t.shape)[1]
                    self.no_of_rows=list(t.shape)[0]
                    self.column_names=json.dumps(list(t.names))
            super(Traindata, self).save(*args, **kwargs)


class Experiment(models.Model):

    class Meta:
        ordering = ['-created_on_date']
    EXPERIMENT_TYPE = [
        ('input', 'Input'),
        ("columnformatchange",'Column format change'),
        ('featureengineering','Feature engineering'),
        ('stationarity', 'Stationarity test'),
        ('manualvariableselection','Manual variable selection'),
        ('classificationmodel','Build logistic regression model')
    ]
    STATUS_TYPE= [("STARTED","STARTED"),("IN_PROGRESS","IN PROGRESS"),("DONE","DONE"),("ERROR","ERROR")]
    experiment_id= models.AutoField(primary_key=True)
    experiment_type=models.CharField(max_length=100,choices=EXPERIMENT_TYPE,null=True, blank=True,default="input")
    name = models.CharField(max_length=100,null=True, blank=True)
    artefacts_directory = models.CharField(max_length=100,null=True, blank=True)
    # start_date = models.DateField(null=True, blank=True)
    created_on_date= models.DateTimeField(auto_now_add=True, null=True, blank=True)
    experiment_status=models.CharField(max_length=20,choices=STATUS_TYPE,null=True, blank=True)
    traindata = models.ForeignKey(Traindata, on_delete=models.CASCADE, null=True,blank=True, related_name ='input_train_data')
    do_create_data =models.BooleanField(default=True)
    enable_spark=models.BooleanField(default=True)
    output_data = models.ForeignKey(Traindata, on_delete=models.CASCADE, null=True,blank=True,related_name ='output_data')
    previous_experiment = models.ForeignKey('self',null=True,blank=True,on_delete=models.CASCADE)
    run_now=models.BooleanField(default=False)
    run_start_time= models.DateTimeField( null=True, blank=True)
    run_end_time= models.DateTimeField(null=True, blank=True)
    created_by = models.ForeignKey(User,
        null=True, blank=True, on_delete=models.CASCADE)

    def __str__(self):
        return self.name if self.name else ''

    def get_fields(self):
        return [(field.verbose_name, field.value_from_object(self)) for field in self.__class__._meta.fields[1:]] 
        
    def get_absolute_url(self):
        return reverse('experiment_detail', args=[str(self.experiment_id)])
    def save(self, *args, **kwargs):
            # if not obj.pk:
            #     # Only set added_by during the first save.
            #     obj.added_by = request.user
            # super().save_model(request, obj, form, change)
            super(Experiment, self).save(*args, **kwargs)
            if not self.artefacts_directory:
                self.artefacts_directory =os.path.join("artefacts","experiment_"+str(self.experiment_id)+"_"+self.name)
                os.makedirs(self.artefacts_directory,exist_ok=True)
                kwargs['force_insert']=False
                super(Experiment, self).save(*args, **kwargs)

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
        output_dir=os.path.join(self.artefacts_directory,'output')


        os.makedirs(output_dir,exist_ok=True)
        files = glob.glob(output_dir+"/*") # look for files inside the directory
        if len(files)>0:
            for f in files:
                os.remove(f)
        if not self.enable_spark:
            if self.run_now:
                if self.do_create_data:
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
        else:
            if self.run_now:
                if self.do_create_data:
                    cols_to_keep=json.loads(self.keep_columns)
                    spark=get_spark_session()
                    df = spark.read.option("header",True).csv(self.traindata.train_path, inferSchema=True)
                    spark_df_subset = df.select(*cols_to_keep)
                    spark_df_subset.write.mode('overwrite').option("header",True).csv(output_dir)
                    no_of_rows = spark_df_subset.count()
                    no_of_cols = len(spark_df_subset.columns)
                    column_names = json.dumps(spark_df_subset.columns)
                    macro_file_obj=Traindata.objects.create(train_path = output_dir, 
                    train_data_name=self.name + "_output",no_of_cols=no_of_cols,no_of_rows=no_of_rows,column_names=column_names)
                    self.output_data=macro_file_obj
                    self.experiment_status="DONE"
                    # spark.read.csv(output_dir)
        # os.makedirs("output",exist_ok=True)
class ClassificationMetrics(models.Model):
    type=models.CharField(max_length=100,blank=True,null=True)
    FPR=models.TextField(max_length=20000,blank=True,null=True)
    TPR=models.TextField(max_length=20000,blank=True,null=True)
    areaUnderROC=models.FloatField(blank=True,null=True)
    precision=models.TextField(max_length=20000,blank=True,null=True)
    recall=models.TextField(max_length=20000,blank=True,null=True)
    thresholds=models.TextField(max_length=20000,blank=True,null=True)
    areaUnderPR=models.FloatField(blank=True,null=True)
    experiment_id=models.IntegerField(blank=True,null=True)

class ResultsClassificationmodel(models.Model):
    coefficients=models.TextField(max_length=20000,blank=True,null=True)
    feature_cols=models.TextField(max_length=20000,blank=True,null=True)
    train_results =models.ForeignKey(ClassificationMetrics, on_delete=models.CASCADE, null=True,blank=True,related_name='train_results')
    test_results =models.ForeignKey(ClassificationMetrics, on_delete=models.CASCADE, null=True,blank=True,related_name='test_results')
class Classificationmodel(Experiment):
    label_col=models.CharField(max_length=100,blank=True,null=True)
    feature_cols=models.TextField(max_length=20000,blank=True,null=True)
    train_split=models.FloatField(blank=True,null=True)
    test_split=models.FloatField(blank=True,null=True)
    feature_cols=models.TextField(max_length=20000,blank=True,null=True)
    ignored_columns=models.TextField(max_length=20000,blank=True,null=True)
    cross_validation=models.BooleanField(default=False)
    results =models.ForeignKey(ResultsClassificationmodel, on_delete=models.CASCADE, null=True,blank=True)

    def get_absolute_url(self):
        if self.experiment_status:
            return reverse('classificationmodel_detail', args=[str(self.experiment_id)])
        else:
            return reverse('classificationmodel_update', args=[str(self.experiment_id)])
    
    def save(self, *args, **kwargs):
            # self.slug = slugify(self.title)
            if self._state.adding: 
                self.experiment_type='classificationmodel'
                # if self.previous_experiment and self.previous_experiment.output_data:
                #     self.traindata=self.previous_experiment.output_data
                #     self.input_columns=json.dumps(pd.read_csv(self.traindata.train_path,nrows=10).columns.to_list())

                #     if  self.run_now:
                #         self.run_start_time= timezone.now()
                #         self.create_output_data()
                #         self.run_end_time= timezone.now()
            else:
                if  self.run_now:
                        self.run_start_time= timezone.now()
                        # self.create_output_data()
                        # from .tasks import run_logistic_regression # this is done to avoid 
                        import logistic_build.tasks as t
                        self.experiment_status='STARTED'
                        super(Classificationmodel, self).save(*args, **kwargs)
                        # async_task("logistic_build.tasks.do_stationarity_test_django_q", self.experiment_id)
                        logistic_results= t.run_logistic_regression(self.experiment_id)
                        train_results=ClassificationMetrics.objects.create(**logistic_results.train_result.all_attributes)
                        test_results=ClassificationMetrics.objects.create(**logistic_results.test_result.all_attributes)

                        self.results=ResultsClassificationmodel.objects.create(train_results=train_results,test_results=test_results,
                                    coefficients=json.dumps(logistic_results.overall_result.coefficients),
                                    feature_cols= json.dumps(logistic_results.overall_result.feature_cols))
                        
            # if self.experiment_status and self.experiment_status=='DONE':
            #     self.create_variables_with_stationarity_results()
            #     if self.do_create_data:
            #         self.create_output_data()

                    
                        self.experiment_status='DONE'
                        self.run_end_time= timezone.now()
                        super(Classificationmodel, self).save(*args, **kwargs)
                        experiment=Experiment.objects.get(pk=self.experiment_id)
                        notification=NotificationModelBuild.objects.create(is_read=False,message='Experiment Successful',experiment=experiment,created_by=experiment.created_by)
                        # Notification.create 
                # df=self.subset_data()
            super(Classificationmodel, self).save(*args, **kwargs)

class NotificationModelBuildManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(is_read=False)

class NotificationModelBuild(models.Model):
    is_read = models.BooleanField(default=False)
    message = models.TextField()
    experiment=models.ForeignKey(Experiment, on_delete=models.CASCADE, null=True,blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    # user = models.ForeignKey(User, on_delete=models.CASCADE)
    objects = NotificationModelBuildManager()
    created_by = models.ForeignKey(User,null=True, blank=True, on_delete=models.CASCADE)


import django_filters

class ExperimentFilter(django_filters.FilterSet):
    name = django_filters.CharFilter(lookup_expr='iexact')
    experiment_status = django_filters.ChoiceFilter(choices=Experiment.STATUS_TYPE,widget=django_filters.widgets.LinkWidget)
    experiment_type = django_filters.ChoiceFilter(choices=Experiment.EXPERIMENT_TYPE,widget=django_filters.widgets.LinkWidget)
    # created_on_date= django_filters.DateFromToRangeFilter(widget=django_filters.widgets.RangeWidget(attrs={'placeholder': 'YYYY/MM/DD'}))

    #     start = django_filters.DateFilter(
    #     field_name='trademonthlystat__last_date', 
    #     lookup_expr='gt',         
    # )
    
    # end = django_filters.DateFilter(
    #     field_name='trademonthlystat__last_date', 
    #     lookup_expr='lt',        
    # )
    # experiment_status = django_filters.CharFilter(lookup_expr='iexact')
    created_on_date__gt = django_filters.NumberFilter(field_name='created_on_date', lookup_expr='year__gt',label="Start")
    created_on_date__lt = django_filters.NumberFilter(field_name='created_on_date', lookup_expr='year__lt', label='End')

    class Meta:
        model = Experiment
        fields = ['experiment_type', 'name','experiment_status']
        # fields = {
        #     'experiment_type': [ 'contains'],
        #     'name': ['contains'],
        #     'created_on_date': ['exact', 'year__gt'],
        # }