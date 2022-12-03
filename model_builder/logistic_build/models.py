from django.db import models
import json

from django.urls import reverse
# Create your models here.


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
    
    traindata = models.ForeignKey(Traindata, on_delete=models.SET_NULL, null=True)
    

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
    var_id= models.AutoField(primary_key=True)
    var_name=models.CharField(max_length=100,null=True, blank=True)
    file_id= models.ForeignKey(Traindata, on_delete=models.SET_NULL, null=True)
    experiment_id= models.ForeignKey(Experiment, on_delete=models.SET_NULL, null=True)
    variable_type= models.CharField(
        max_length=20,
        choices=MODEL_VARIABLE_TYPE,
        default='Independent',
    )