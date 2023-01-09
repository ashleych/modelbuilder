import django_filters
from django.db import models
import json

from django.urls import reverse

from .utils import get_spark_session
# Create your models here.
from .stationarity_tests import adfuller_test, kpss_test
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
    file_id = models.AutoField(primary_key=True)
    train_data_name = models.CharField(max_length=100, null=True, blank=True)
    train_path = models.CharField(max_length=50)
    no_of_cols = models.IntegerField(null=True, blank=True)
    no_of_rows = models.IntegerField(null=True, blank=True)
    column_names = models.CharField(max_length=50, null=True, blank=True)
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
                t = dt.fread(file=self.train_path)
                self.no_of_cols = list(t.shape)[1]
                self.no_of_rows = list(t.shape)[0]
                self.column_names = json.dumps(list(t.names))
        super(Traindata, self).save(*args, **kwargs)


class Experiment(models.Model):

    class Meta:
        ordering = ['-created_on_date']
    EXPERIMENT_TYPE = [
        ('input', 'Input'),
        ("columnformatchange", 'Column format change'),
        ('featureengineering', 'Feature engineering'),
        ('stationarity', 'Stationarity test'),
        ('manualvariableselection', 'Manual variable selection'),
        ('featureselection', 'Feature Selection'),
        ('classificationmodel', 'Build logistic regression model'),
        ('regressionmodel', 'Build regression model')
    ]
    STATUS_TYPE = [("NOT_STARTED", "NOT STARTED"), ("STARTED", "STARTED"), ("IN_PROGRESS", "IN PROGRESS"), ("DONE", "DONE"), ("ERROR", "ERROR")]
    experiment_id = models.AutoField(primary_key=True)
    experiment_type = models.CharField(max_length=100, choices=EXPERIMENT_TYPE, null=True, blank=True, default="input")
    name = models.CharField(max_length=100)
    artefacts_directory = models.CharField(max_length=100, null=True, blank=True)
    # start_date = models.DateField(null=True, blank=True)
    created_on_date = models.DateTimeField(auto_now_add=True, null=True, blank=True)
    experiment_status = models.CharField(max_length=20, choices=STATUS_TYPE, null=True, blank=True)
    traindata = models.ForeignKey(Traindata, on_delete=models.CASCADE, null=True, blank=True, related_name='input_train_data')
    do_create_data = models.BooleanField(default=True)
    output_data = models.ForeignKey(Traindata, on_delete=models.CASCADE, null=True, blank=True, related_name='output_data')
    previous_experiment = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)
    run_now = models.BooleanField(default=False)
    lock_now = models.BooleanField(default=False)
    run_start_time = models.DateTimeField(null=True, blank=True)
    run_end_time = models.DateTimeField(null=True, blank=True)
    created_by = models.ForeignKey(User,
                                   null=True, blank=True, on_delete=models.CASCADE)
    all_preceding_experiments = models.TextField(max_length=20000, blank=True, null=True)
    run_in_the_background = models.BooleanField(default=False)
    enable_spark = models.BooleanField(default=False)

    def _track_precedents(self):
        if self.previous_experiment:
            if self.previous_experiment.all_preceding_experiments:
                prev_exp_dict = json.loads(self.previous_experiment.all_preceding_experiments)
                prev_exp_dict.append((self.previous_experiment.experiment_id, self.previous_experiment.experiment_type, self.previous_experiment.name))
                # prev_exp_dict[self.experiment_id]= f"{self.experiment_type}:{self.exeriment_name}"
                self.all_preceding_experiments = json.dumps(prev_exp_dict)
            else:
                prev_exp_dict = [(self.previous_experiment.experiment_id, self.previous_experiment.experiment_type, self.previous_experiment.name)]
                self.all_preceding_experiments = json.dumps(prev_exp_dict)

    def __str__(self):
        return self.name if self.name else ''

    def get_fields(self):
        return [(field.verbose_name, field.value_from_object(self)) for field in self.__class__._meta.fields[1:]]

    def get_absolute_url(self):
        if self.experiment_type != 'input':
            return reverse(f'{self.experiment_type}_detail', args=[str(self.experiment_id)])
        else:
            return reverse('experiment_detail', args=[str(self.experiment_id)])

    def save(self, *args, **kwargs):
        print(kwargs)
        self._track_precedents()
        super(Experiment, self).save(*args, **kwargs)
        # super(Experiment, self).save(*args, **kwargs)
        if not self.artefacts_directory:
            self.artefacts_directory = os.path.join("artefacts", "experiment_"+str(self.experiment_id)+"_"+self.name)
            os.makedirs(self.artefacts_directory, exist_ok=True)
            kwargs['force_insert'] = False
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
    variable_id = models.AutoField(primary_key=True)
    variable_name = models.CharField(max_length=100, null=True, blank=True)
    traindata = models.ForeignKey(Traindata, on_delete=models.SET_NULL, blank=True, null=True)
    experiment = models.ForeignKey(Experiment, on_delete=models.SET_NULL, null=True)
    variable_type = models.CharField(
        max_length=20,
        choices=MODEL_VARIABLE_TYPE,
        default='Independent',
    )
    experiment_status = models.CharField(
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
    do_kpss = models.BooleanField(default=False)
    do_adf = models.BooleanField(default=False)
    significance = models.FloatField(null=True, blank=True, default=0.05)
    kpss_pass_vars = models.TextField(max_length=20000, blank=True, null=True)
    adf_pass_vars = models.TextField(max_length=20000, blank=True, null=True)
    stationarity_passed = models.TextField(max_length=20000, blank=True, null=True)

    def set_json(self, field, x):
        listVal = json.dumps(list(x))
        setattr(self, field, listVal)

    def get_json(self, field):
        return json.loads(getattr(self, field))

    def updates_variable_list(self):
        pass

    def create_variables_with_stationarity_results(self):
        if self.experiment_status:
            for col in json.loads(self.stationarity_passed):
                Variables.objects.create(variable_name=col, traindata=self.traindata, experiment=self, experiment_status="PASS")

    def create_output_data(self, *args, **kwargs):
        # import pdb; pdb.set_trace()
        # The path exists check is needed cos the background task do_stationary_tests calls this function as well whose cwd is different
        # ideally it should not be, but till the time the issue is fixed, one has to check if the path exists, else to go up one directory (whcih is where the tasks. py initiated workers have as directory)
        # essentially its whether the path is model_builder or model_builder/model_builder
        if os.path.exists(self.traindata.train_path):
            print("path exists " + self.traindata.train_path)
            file_path = self.traindata.train_path
            file_dir = ""
        else:
            # print( "path exists "+ self.traindata.train_path)
            file_path = os.path.join(Path(settings.BASE_DIR).parent, self.traindata.train_path)
            file_dir = os.path.join(Path(settings.BASE_DIR).parent)
            print("path exists " + file_path)
            print(file_path)
        input_data = pd.read_csv(file_path)
        subset = input_data[json.loads(self.stationarity_passed)]
        out_dir_path = os.path.join(file_dir, "output")
        os.makedirs(out_dir_path, exist_ok=True)
        file_name_as_stored_on_disk = os.path.join(out_dir_path, self.name+"_"+"Exp_id_" + str(self.experiment_id) + ".csv")
        if os.path.exists(file_name_as_stored_on_disk):
            os.remove(file_name_as_stored_on_disk)
        print(file_name_as_stored_on_disk+" :    file_namesa as stpred ")
        # portfolio_data_input= default_storage.save(os.path.join("input","input.csv"), csv_file)
        subset.to_csv(os.path.join(file_name_as_stored_on_disk))
        macro_file_obj = Traindata.objects.create(train_path=file_name_as_stored_on_disk, train_data_name=self.name + "_output")
        self.output_data = macro_file_obj

    def save(self, *args, **kwargs):
        if self._state.adding:
            self.experiment_type = 'stationarity'
            # self.experiment_status= 'NOT STARTED'
            super(Stationarity, self).save(*args, **kwargs)
        # self.experiment_type='stationarity'
        # super(Stationarity, self).save(*args, **kwargs)
        if self.run_now:
            self.run_end_time = None
            self.run_start_time = timezone.now()
            self.results = None
            # from .tasks import run_logistic_regression # this is done to avoid
            import logistic_build.tasks as t
            self.experiment_status = 'IN_PROGRESS'
            super(Stationarity, self).save(*args, **kwargs)
            if self.run_in_the_background:
                async_task("logistic_build.tasks.do_stationarity_test_django_q", self.experiment_id)
            else:
                result = t.do_stationarity_test_django_q(self.experiment_id)
        else:
            if self.experiment_status == 'DONE' or self.experiment_status == 'NOT_STARTED':
                super(Stationarity, self).save(*args, **kwargs)

    def get_absolute_url(self):
        if self.experiment_status:
            return reverse('stationarity_detail', args=[str(self.experiment_id)])
        else:
            return reverse('stationarity_update', args=[str(self.experiment_id)])


class Manualvariableselection(Experiment):

    input_columns = models.TextField(max_length=20000, blank=True, null=True)
    keep_columns = models.TextField(max_length=20000, blank=True, null=True)

    def save(self, *args, **kwargs):
        # self.slug = slugify(self.title)
        if self._state.adding:
            self.experiment_type = 'manualvariableselection'
            if self.previous_experiment and self.previous_experiment.output_data:
                self.traindata = self.previous_experiment.output_data
                self.input_columns = json.dumps(pd.read_csv(self.traindata.train_path, nrows=10).columns.to_list())

                if self.run_now:
                    self.run_start_time = timezone.now()
                    self.create_output_data()
                    self.run_end_time = timezone.now()
        else:
            if self.run_now:
                self.run_start_time = timezone.now()
                self.create_output_data()
                self.run_end_time = timezone.now()
        super(Manualvariableselection, self).save(*args, **kwargs)
        # if self.do_create_data:
        #         self.create_output_data(df)

    def get_absolute_url(self):
        if self.experiment_status:
            return reverse('manualvariableselection_detail', args=[str(self.experiment_id)])
        else:
            return reverse('manualvariableselection_update', args=[str(self.experiment_id)])

    def create_output_data(self):
        output_dir = os.path.join(self.artefacts_directory, 'output')

        os.makedirs(output_dir, exist_ok=True)
        files = glob.glob(output_dir+"/*")  # look for files inside the directory
        if len(files) > 0:
            for f in files:
                os.remove(f)
        if not self.enable_spark:
            if self.run_now:
                if self.do_create_data:
                    cols_to_keep = json.loads(self.keep_columns)
                    input_data = pd.read_csv(self.traindata.train_path)
                    output_data = input_data[cols_to_keep]

                    file_name_as_stored_on_disk = os.path.join("output", self.name+"_"+"Exp_id_" + str(self.experiment_id) + ".csv")
                    if os.path.exists(file_name_as_stored_on_disk):
                        os.remove(file_name_as_stored_on_disk)
                    output_data.to_csv(file_name_as_stored_on_disk)
                    macro_file_obj = Traindata.objects.create(train_path=file_name_as_stored_on_disk, train_data_name=self.name + "_output")
                    self.output_data = macro_file_obj
                    self.experiment_status = "DONE"
        else:
            if self.run_now:
                if self.do_create_data:
                    cols_to_keep = json.loads(self.keep_columns)
                    spark = get_spark_session()
                    df = spark.read.option("header", True).csv(self.traindata.train_path, inferSchema=True)
                    spark_df_subset = df.select(*cols_to_keep)
                    spark_df_subset.write.mode('overwrite').option("header", True).csv(output_dir)
                    no_of_rows = spark_df_subset.count()
                    no_of_cols = len(spark_df_subset.columns)
                    column_names = json.dumps(spark_df_subset.columns)
                    macro_file_obj = Traindata.objects.create(train_path=output_dir,
                                                              train_data_name=self.name + "_output", no_of_cols=no_of_cols, no_of_rows=no_of_rows, column_names=column_names)
                    self.output_data = macro_file_obj
                    self.experiment_status = "DONE"


class ClassificationMetrics(models.Model):
    type = models.CharField(max_length=100, blank=True, null=True)
    FPR = models.TextField(max_length=20000, blank=True, null=True)
    TPR = models.TextField(max_length=20000, blank=True, null=True)
    areaUnderROC = models.FloatField(blank=True, null=True)
    fp = models.FloatField(blank=True, null=True)
    tp = models.FloatField(blank=True, null=True)
    fn = models.FloatField(blank=True, null=True)
    tn = models.FloatField(blank=True, null=True)
    precision = models.FloatField(blank=True, null=True)
    recall = models.FloatField(blank=True, null=True)

    accuracy = models.FloatField(blank=True, null=True)
    npv = models.FloatField(blank=True, null=True)
    precision_plot_data = models.TextField(max_length=20000, blank=True, null=True)
    recall_plot_data = models.TextField(max_length=20000, blank=True, null=True)
    pr_thresholds = models.TextField(max_length=20000, blank=True, null=True)
    f1_score = models.FloatField(blank=True, null=True)
    areaUnderPR = models.FloatField(blank=True, null=True)
    specificity = models.FloatField(blank=True, null=True)
    experiment_id = models.IntegerField(blank=True, null=True)


class ResultsClassificationmodel(models.Model):
    coefficients = models.TextField(max_length=20000, blank=True, null=True)
    features = models.TextField(max_length=20000, blank=True, null=True)
    train_nrows = models.IntegerField(blank=True, null=True)
    test_nrows = models.IntegerField(blank=True, null=True)
    train_results = models.ForeignKey(ClassificationMetrics, on_delete=models.CASCADE, null=True, blank=True, related_name='train_results')
    test_results = models.ForeignKey(ClassificationMetrics, on_delete=models.CASCADE, null=True, blank=True, related_name='test_results')


class Classificationmodel(Experiment):
    label_col = models.CharField(max_length=100, blank=True, null=True)
    feature_cols = models.TextField(max_length=20000, blank=True, null=True)
    train_split = models.FloatField(blank=True, null=True)
    test_split = models.FloatField(blank=True, null=True)
    feature_cols = models.TextField(max_length=20000, blank=True, null=True)
    ignored_columns = models.TextField(max_length=20000, blank=True, null=True)
    cross_validation = models.IntegerField(default=0, null=True, blank=True)
    results = models.ForeignKey(ResultsClassificationmodel, on_delete=models.CASCADE, null=True, blank=True)

    def get_absolute_url(self):
        if self.experiment_status and self.experiment_status != 'NOT_STARTED':
            return reverse('classificationmodel_detail', args=[str(self.experiment_id)])
        else:
            return reverse('classificationmodel_update', args=[str(self.experiment_id)])

    def save(self, *args, **kwargs):
        if self._state.adding:
            self.experiment_type = 'classificationmodel'
            self.experiment_status = 'NOT_STARTED'
            super(Classificationmodel, self).save(*args, **kwargs)

        if self.run_now and self.traindata and self.label_col:
            self.run_end_time = None
            self.run_start_time = timezone.now()
            self.results = None
            # from .tasks import run_logistic_regression # this is done to avoid
            import logistic_build.tasks as t
            self.experiment_status = 'STARTED'
            kwargs['force_insert'] = False
            super(Classificationmodel, self).save(*args, **kwargs)
            if self.run_in_the_background:
                async_task("logistic_build.tasks.run_logistic_regression", self.experiment_id, run_in_the_background=True)
                # logistic_results= t.run_logistic_regression(self.experiment_id,run_in_the_background=True) # to run in foreground, for checks
            else:
                logistic_results = t.run_logistic_regression(self.experiment_id)
        else:
            print("Experiment status :", self.experiment_status)
            if self.experiment_status == 'DONE':
                kwargs['force_insert'] = False
                super(Classificationmodel, self).save(*args, **kwargs)
            else:
                if self.experiment_status == 'NOT_STARTED':
                    self.run_end_time = None
                    self.run_start_time = timezone.now()
                    self.results = None
                    kwargs['force_insert'] = False
                    super(Classificationmodel, self).save(*args, **kwargs)


class RegressionMetrics(models.Model):
    type = models.CharField(max_length=100, blank=True, null=True)
    mae = models.FloatField(blank=True, null=True)
    r_squared = models.FloatField(blank=True, null=True)
    mse = models.FloatField(blank=True, null=True)
    rmse = models.FloatField(blank=True, null=True)
    mae = models.FloatField(blank=True, null=True)
    explained_variance = models.FloatField(blank=True, null=True)
    experiment_id = models.IntegerField(blank=True, null=True)


class ResultsRegressionmodel(models.Model):
    coefficients = models.TextField(max_length=20000, blank=True, null=True)
    intercept = models.FloatField(blank=True, null=True)
    feature_cols = models.TextField(max_length=20000, blank=True, null=True)
    tvalues = models.TextField(max_length=20000, blank=True, null=True)
    pvalues = models.TextField(max_length=20000, blank=True, null=True)
    dispersion = models.FloatField(blank=True, null=True)
    coefficientsStandardErrors = models.TextField(max_length=20000, blank=True, null=True)
    nullDeviance = models.FloatField(blank=True, null=True)
    residualDegreeOfFreedomNull = models.FloatField(blank=True, null=True)
    residualDegreeOfFreedom = models.FloatField(blank=True, null=True)
    deviance = models.FloatField(blank=True, null=True)
    AIC = models.FloatField(blank=True, null=True)
    train_results = models.ForeignKey(RegressionMetrics, on_delete=models.CASCADE, null=True, blank=True, related_name='train_results')
    test_results = models.ForeignKey(RegressionMetrics, on_delete=models.CASCADE, null=True, blank=True, related_name='test_results')


class Regressionmodel(Experiment):
    label_col = models.CharField(max_length=100, blank=True, null=True)
    feature_cols = models.TextField(max_length=20000, blank=True, null=True)
    train_split = models.FloatField(blank=True, null=True)
    test_split = models.FloatField(blank=True, null=True)
    feature_cols = models.TextField(max_length=20000, blank=True, null=True)
    ignored_columns = models.TextField(max_length=20000, blank=True, null=True)
    cross_validation = models.BooleanField(default=False)
    results = models.ForeignKey(ResultsRegressionmodel, on_delete=models.CASCADE, null=True, blank=True)

    def get_absolute_url(self):
        if self.experiment_status and self.experiment_status != 'NOT_STARTED':
            return reverse('regressionmodel_detail', args=[str(self.experiment_id)])
        else:
            return reverse('regressionmodel_update', args=[str(self.experiment_id)])

    def save(self, *args, **kwargs):
        print(f"self run now : {self.run_now}")
        if self._state.adding:
            self.experiment_type = 'regressionmodel'
            self.experiment_status = 'NOT_STARTED'
            super(Regressionmodel, self).save(*args, **kwargs)

        else:
            if self.run_now:
                self.run_end_time = None
                self.run_start_time = timezone.now()
                self.results = None
                # from .tasks import run_logistic_regression # this is done to avoid
                import logistic_build.tasks as t
                self.experiment_status = 'STARTED'
                super(Regressionmodel, self).save(*args, **kwargs)

                if self.run_in_the_background:
                    async_task("logistic_build.tasks.run_glm_regression", self.experiment_id, run_in_the_background=True)
                else:
                    t.run_glm_regression(self.experiment_id)
            else:
                print("Experiment status :", self.experiment_status)
                if self.experiment_status == 'DONE':
                    super(Regressionmodel, self).save(*args, **kwargs)
                else:
                    if self.experiment_status == 'NOT_STARTED':
                        self.run_end_time = None
                        self.run_start_time = timezone.now()
                        self.results = None
                        super(Regressionmodel, self).save(*args, **kwargs)


class NotificationModelBuildManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset()
        # return super().get_queryset().filter(is_read=False)


class NotificationModelBuild(models.Model):
    is_read = models.BooleanField(default=False)
    message = models.TextField()
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE, null=True, blank=True)
    experiment_type = models.CharField(max_length=100, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    # user = models.ForeignKey(User, on_delete=models.CASCADE)
    objects = NotificationModelBuildManager()
    created_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.CASCADE)

    class Meta:
        ordering = ['-timestamp']


class ExperimentFilter(django_filters.FilterSet):
    # This is used to power the filters to the right of the all_experiments view
    name = django_filters.CharFilter(lookup_expr='icontains')
    experiment_status = django_filters.ChoiceFilter(choices=Experiment.STATUS_TYPE, widget=django_filters.widgets.LinkWidget)
    experiment_type = django_filters.ChoiceFilter(choices=Experiment.EXPERIMENT_TYPE, widget=django_filters.widgets.LinkWidget)
    created_on_date__gt = django_filters.NumberFilter(field_name='created_on_date', lookup_expr='year__gt', label="Start")
    created_on_date__lt = django_filters.NumberFilter(field_name='created_on_date', lookup_expr='year__lt', label='End')

    class Meta:
        model = Experiment
        fields = ['experiment_type', 'name', 'experiment_status']
# class ResultsFeatureselectionManager(models.Manager):
#     def get_queryset(self):
#         s=super().get_queryset()
#         return s
        # return super().get_queryset().filter(is_read=False)

class ResultsFeatureselection(models.Model):
    constant_features = models.TextField(max_length=20000, blank=True, null=True)
    quasi_constant_features = models.TextField(max_length=20000, blank=True, null=True)
    correlated_features = models.TextField(max_length=20000, blank=True, null=True)
    duplicated_features = models.TextField(max_length=20000, blank=True, null=True)
    non_numeric_columns = models.TextField(max_length=20000, blank=True, null=True)
    shortlisted_features = models.TextField(max_length=20000, blank=True, null=True)
    # objects = ResultsFeatureselectionManager()
    # featureselection = models.ForeignKey(Featureselection, null=True, blank=True, on_delete=models.CASCADE)

    def save(self, *args, **kwargs):
        string_attributes = ["constant_features", "quasi_constant_features", "correlated_features", "duplicated_features", "non_numeric_columns", "shortlisted_features"]
        for attrib in string_attributes:
            value_ = getattr(self, attrib)
            if type(value_) == list:
                if value_:
                    setattr(self, attrib, json.dumps(value_))
                else:
                    setattr(self, attrib, None)
        super(ResultsFeatureselection, self).save(*args, **kwargs)

    # def __str__(self):
    #     return self.train_data_name if self.train_data_name else ''

    def get_fields(self):
        return [(field.verbose_name, field.value_from_object(self)) for field in self.__class__._meta.fields[1:]]

class TopModelsManager():

    def get_queryset(self):
        return super().get_queryset()
class TopModels(models.Model):
    selected_features = models.TextField(max_length=20000, blank=True, null=True)
    cv_scores = models.TextField(max_length=20000, blank=True, null=True)
    avg_score = models.FloatField(blank=True, null=True)
    rank = models.IntegerField(blank=True, null=True)
    results = models.ForeignKey(ResultsFeatureselection, null=True, blank=True, on_delete=models.CASCADE)
    topmodels = TopModelsManager()

    def save(self, *args, **kwargs):
        string_attributes = ["selected_features", "cv_scores"]
        for attrib in string_attributes:
            value_ = getattr(self, attrib)
            if type(value_) == list:
                if value_:
                    setattr(self, attrib, json.dumps(value_))
                else:
                    setattr(self, attrib, None)
        super(TopModels, self).save(*args, **kwargs)

class Featureselection(Experiment):

    TASK_TYPE = [('regression', 'Regression'), ('classification', 'Classification')]
    SCORING_TYPE = [('roc_auc', 'ROC AUC'), ('mse', 'MSE')]
    label_col = models.CharField(max_length=100, blank=True, null=True)
    feature_cols = models.TextField(max_length=20000, blank=True, null=True)
    train_split = models.FloatField(blank=True, null=True)
    test_split = models.FloatField(blank=True, null=True)
    exclude_features = models.TextField(max_length=20000, blank=True, null=True)
    fixed_features = models.TextField(max_length=20000, blank=True, null=True)
    cross_validation = models.IntegerField(default=5, null=True, blank=True)
    results = models.ForeignKey(ResultsFeatureselection, on_delete=models.CASCADE, null=True, blank=True)
    max_features = models.IntegerField(default=5, blank=True, null=True)
    min_features = models.IntegerField(default=2, blank=True, null=True)
    short_list_max_features = models.IntegerField(default=5, blank=True, null=True)
    regression_or_classification = models.CharField(max_length=20, choices=TASK_TYPE, default='regression')
    scoring = models.CharField(max_length=10, choices=SCORING_TYPE, default='roc_auc')
    remove_constant_features = models.BooleanField(default=False, null=True, blank=True)
    remove_quasi_constant_features = models.BooleanField(default=False, null=True, blank=True)
    variance_threshold = models.BooleanField(default=False, null=True, blank=True)  # for quasi constant check
    correlation_check = models.BooleanField(default=False, null=True, blank=True)
    correlation_threshold = models.FloatField(blank=True, null=True)
    treat_missing = models.BooleanField(default=False, null=True, blank=True)
    variables_selected = models.BooleanField(default=False, null=True, blank=True)
    do_exhaustive_search = models.BooleanField(default=False, null=True, blank=True)

    def get_absolute_url(self):
        if self.experiment_status and self.experiment_status != 'NOT_STARTED':
            return reverse('featureselection_detail', args=[str(self.experiment_id)])
        else:
            return reverse('featureselection_update', args=[str(self.experiment_id)])

    def save(self, *args, **kwargs):
        print(f"self run now : {self.run_now}")
        if self._state.adding:
            self.experiment_type = 'featureselection'
            self.experiment_status = 'NOT_STARTED'
            super(Featureselection, self).save(*args, **kwargs)

        else:
            if self.run_now:
                self.run_end_time = None
                self.run_start_time = timezone.now()
                self.results = None
                import logistic_build.tasks as task_runner
                self.experiment_status = 'STARTED'
                super(Featureselection, self).save(*args, **kwargs)

                if self.run_in_the_background:
                    async_task("logistic_build.tasks.run_feature_selection", self.experiment_id)
                else:
                    task_runner.run_feature_selection(self.experiment_id)
            else:
                print("Experiment status :", self.experiment_status)
                if self.experiment_status == 'DONE':
                    super(Featureselection, self).save(*args, **kwargs)
                else:
                    if self.experiment_status == 'NOT_STARTED':
                        self.run_end_time = None
                        self.run_start_time = timezone.now()
                        self.results = None
                        super(Featureselection, self).save(*args, **kwargs)
