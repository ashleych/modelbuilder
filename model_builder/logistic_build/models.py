from django.db import models
import json
# Create your models here.


class Traindata(models.Model):
    file_id= models.AutoField(primary_key=True)
    train_path = models.CharField(max_length=50)
    colnames = models.CharField(max_length=54)
    relevant_col_names = models.CharField(max_length=500)

    def set_relevant_col_names(self, x):
        self.relevant_col_names = json.dumps(x)

    def get_relevant_col_names(self):
        return json.loads(self.relevant_col_names)

    def get_col_names(self):
        return json.loads(self.colnames)


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


