#%% 
from sklearn.metrics import precision_recall_curve
from typing import (TypeVar, Generic, Iterable, Optional)
import pyspark
from sklearn.metrics import precision_recall_curve
from typing import (TypeVar, Generic, Iterable, Optional)
import pyspark
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from typing import List
from pyspark.ml.classification import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import os

from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot as plt 
import plotly.express as px
from pyspark.ml.regression import GeneralizedLinearRegression
from typing import List
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot as plt 
import plotly.express as px
from pathlib import Path
from dataclasses import dataclass
import json

from django.conf import settings

def auto_str(cls):
# https://stackoverflow.com/questions/32910096/is-there-a-way-to-auto-generate-a-str-implementation-in-python
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    return cls

@auto_str
class RegressionMetrics():
# https://www.kaggle.com/code/solaznog/mllib-spark-and-pyspark
    def __init__(self, type) -> None:
        self.mae=None
        self.r_squared=None
        self.mse=None
        self.rmse=None
        self.mae=None
        self.explained_variance = None
        self.type=type
      
    @property
    def all_attributes(self):
        all_attrib=dict(vars(self))
        keys= list(all_attrib.keys()) 
        for key in keys:
          if key.startswith("_"): #remove all private keys
            all_attrib.pop(key, 'No Key found')

        return all_attrib
@auto_str
class OverallRegressionResults():
    def __init__(self) -> None:
      self.intercept=None
      self._coefficients=None
      self._feature_cols=None
      self._tvalues=None
      self._pvalues=None
      self._coefficientStandardErrors=None
      


    @property
    def coefficients(self):
        return json.loads(self._coefficients)
      
    # a setter function
    @coefficients.setter
    def coefficients(self, a):
        self._coefficients = json.dumps(list(a))

    @property
    def feature_cols(self):
        return json.loads(self._feature_cols)
      
    # a setter function
    @feature_cols.setter
    def feature_cols(self, a):
        self._feature_cols = json.dumps(list(a))

    @property
    def tvalues(self):
        return json.loads(self._tvalues)
      
    # a setter function
    @tvalues.setter
    def tvalues(self, a):
        self._tvalues = json.dumps(list(a))


    @property
    def pvalues(self):
        return json.loads(self._pvalues)
      
    # a setter function
    @pvalues.setter
    def pvalues(self, a):
        self._pvalues = json.dumps(list(a))


    @property
    def coefficientsStandardErrors(self):
        return json.loads(self._coefficientsStandardErrors)
      
    # a setter function
    @coefficientsStandardErrors.setter
    def coefficientsStandardErrors(self, a):
        self._coefficientsStandardErrors = json.dumps(list(a))

    @property
    def all_attributes(self):
        all_attrib=dict(vars(self),coefficients=self.coefficients,feature_cols=self.feature_cols,tvalues=self.tvalues,pvalues=self.pvalues,coefficientsStandardErrors=self.coefficientsStandardErrors)
        keys= list(all_attrib.keys()) 
        for key in keys:
          if key.startswith("_"): #remove all private keys
            all_attrib.pop(key, 'No Key found')
        return all_attrib
class RegressionModel_spark:
  def __init__( self,
                filepath: str = '/content/car data.csv',
                label_col: str = "label",
                selected_columns: List[str] = ['Year', 'Present_Price', 'Kms_Driven', 'Owner'],
                remove_columns: List[str] = [],
                inferSchema: bool = True,
                standardize: bool = False,
                normalize: bool = False,
                train_percent: float = 0.7,
                test_percent: float = 0.3,
                seed: int = 2022,
                family: str = "gaussian",  
                link: str = "identity",
                max_iterations: int = 10,
                regularization_param: float = 0.3
               ):
    
    self.spark  = self.get_spark_session()
    self.selected_cols=selected_columns
    self.train_result = RegressionMetrics(type='train') # Initialise this 
    self.test_result = RegressionMetrics(type='test') # Initialise this 
    self.overall_result=OverallRegressionResults() # Initialise this
    self.df = self.read_csv_data(filepath,inferSchema=inferSchema)
    self.columns = self.df.columns
    self.remove_cols(remove_columns)
    self.label_col= label_col
    self.feature_cols = self.get_feature_columns(label_col)
    self.spark_output_colname=None
    self.stages = []
    self.add_assembler(selected_columns, label_col)
    if normalize:
      self.add_normaliser(label_col)
    self.train, self.test = self.train_test_split(train_percent, test_percent, seed)
    self.build_pipeline()
    self.transform_data()
    self.glrModel= self.build_regression_model(family, link, max_iterations, regularization_param)
    self.predict_test()
    self.evaluate_data()
    self.get_intercept()
    self.get_coefficients()
    self.get_summary()
    self.get_coefficients_names()
    # get all training metrics
    self.train_result.mae= self.get_mae(self.train_pred_results)
    self.train_result.mse= self.get_mse(self.train_pred_results)
    self.train_result.rmse= self.get_rmse(self.train_pred_results)
    self.train_result.r_squared= self.get_rsquared(self.train_pred_results)
    self.train_result.explained_variance= self.get_explainedvariance(self.train_pred_results)
    self.train_residuals_plot_data=self.get_residuals(self.train_pred_results)

    # get all test metrics
    self.test_result.mae= self.get_mae(self.test_pred_results)
    self.test_result.mse= self.get_mse(self.test_pred_results)
    self.test_result.rmse= self.get_rmse(self.test_pred_results)
    self.test_result.r_squared= self.get_rsquared(self.test_pred_results)
    self.test_result.explained_variance= self.get_explainedvariance(self.test_pred_results)
    self.test_residuals_plot_data=self.get_residuals(self.test_pred_results)

  def get_residuals(self,data,max_rows=1000):
      data_sub=data.limit(max_rows)

      data_sub=data_sub.withColumn('residuals', data_sub[self.label_col] - data_sub.prediction)
      # featuresCol = self.spark_output_colname, labelCol = 'label'
      data_pandas = data_sub.toPandas()
      results_dict={}
      results_dict['predicted']=data_pandas.prediction.to_list()
      results_dict['observed']=data_pandas[self.label_col].to_list()
      results_dict['residuals']=data_pandas.residuals.to_list()

      return results_dict


  def get_spark_session(app_name:str = "RegressionTool"):
      spark = SparkSession.builder.appName(app_name).config("spark.memory.offHeap.enabled","true")\
                          .config("spark.memory.offHeap.size","10g").config('spark.executor.memory', '2g')\
                          .config("spark.driver.host","localhost").getOrCreate()
      return spark


  def read_csv_data(self,filepath:str,inferSchema: bool = True):
    if os.path.exists(filepath):
      df = self.spark.read.option("header",True).csv(filepath, inferSchema=inferSchema)
      # print(df.columns)
    else:
      RuntimeError(f"Path {filepath} not found")
    return df   

  def remove_cols(self, remove_cols_list):
      self.df = self.df.drop(*remove_cols_list)

  def get_feature_columns(self, label_col):
      cols = self.df.columns
      if self.selected_cols:
        cols=self.selected_cols
      if label_col in cols:
        cols.remove(label_col)
      return cols  

  # def get_correlations(self):
  #   correlations = self.df.sample(withReplacement=False, fraction=0.2, seed=44).select(self.columns[1:]).toPandas().corr()
  #   print('Correlations above 0.3: ', np.sum(np.sum(abs(correlations)>0.3)) - 201)  

  # need to make this dynamic
  def add_assembler(self, selected_columns: List[str], label_col: str):
    stages = []
    numericCols = self.feature_cols
    assembler = VectorAssembler(inputCols=numericCols, outputCol='features_vector')
    self.stages += [assembler]
    self.spark_output_colname="features_vector"   

  def add_normaliser(self):
    from pyspark.ml.feature import StandardScaler

    standardScaler = StandardScaler()
    standardScaler.setWithMean(True)
    standardScaler.setWithStd(True)
    standardScaler.setInputCol(self.spark_output_colname)
    standardScaler.setOutputCol("features")
    self.spark_output_colname="features"
    self.stages += [standardScaler]

  def train_test_split(self, train_percent, test_percent, seed):
    train, test = self.df.randomSplit([train_percent, test_percent], seed = seed)
    return train, test

  def build_pipeline(self): 
    pipeline = Pipeline(stages = self.stages)
    self.pipelineModel = pipeline.fit(self.train)

  def transform_data(self):
    self.piped_train_data = self.pipelineModel.transform(self.train)
    self.piped_test_data = self.pipelineModel.transform(self.test)

  def build_regression_model(self, family, link, max_iterations, regularization_param):
    glr = GeneralizedLinearRegression(family=family, link=link, maxIter=max_iterations, 
                                      regParam=regularization_param, featuresCol=self.spark_output_colname,
                                      labelCol= self.label_col)  

    # glr = LinearRegression(featuresCol=self.spark_output_colname, labelCol=label_col)

    glrModel = glr.fit(self.piped_train_data)

    return glrModel

  def predict_test(self):
    self.test_pred_results = self.glrModel.transform(self.piped_test_data)
    self.train_pred_results = self.glrModel.transform(self.piped_train_data)

  # def predict_train(self):
  #   self.train_pred_results = self.glrModel.transform(self.piped_train_data)
  

  def evaluate_data(self):
    self.pred_train = self.glrModel.evaluate(self.piped_train_data)
    self.pred_test = self.glrModel.evaluate(self.piped_test_data)  


  def get_summary(self):
    print(self.glrModel.summary)


  def get_mae(self,data):
    mae = RegressionEvaluator( labelCol= self.label_col, predictionCol="prediction",metricName="mae")
    mae = mae.evaluate(data)
    return(mae)
    # print("MAE for test set: ", mae)

  def get_mse(self, data):
    mse = RegressionEvaluator(labelCol= self.label_col, predictionCol="prediction", metricName="mse")
    mse = mse.evaluate(data)
    return(mse)
    # print("MSE for test set: ", mse)

  def get_rmse(self, data):
    rmse = RegressionEvaluator(labelCol= self.label_col, predictionCol="prediction", metricName="rmse")
    return(rmse.evaluate(data))
    # print("RMSE for test set: ", rmse)

  def get_rsquared(self, data):
    r2 = RegressionEvaluator(labelCol= self.label_col, predictionCol="prediction", metricName="r2")
    return(r2.evaluate(data) )
    # print("R-Squared for test set: ", r2)

  def get_explainedvariance(self,data):
    exp_var= RegressionEvaluator(labelCol= self.label_col, predictionCol="prediction", metricName="var")
    return(exp_var.evaluate(data) )
    # print("Explained Variance for test set: ", exp_var)  

  def get_intercept(self):
    self.overall_result.intercept=self.glrModel.intercept
    print('Intercept: ', self.glrModel.intercept)  

  def get_coefficients(self):
    print('Coefficients: ', self.glrModel.coefficients)   

  def view_predictions(self):
    self.pred_train.predictions.show()
    self.pred.predictions.show()  

  def get_coefficients_names(self):
    # Source : https://turingintern2018.github.io/sparkairplane.html
      import numpy as np
      summary = self.glrModel.summary
      modelcoefficients=np.array(self.glrModel.coefficients)
      names_with_idx=[]
      if self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('numeric'):
          names_with_idx += self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('numeric')
      
      if self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('binary'):
          names_with_idx+= self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('binary')
      names=[x['name'] for x in sorted(names_with_idx, key= lambda x:x["idx"])]
      matchcoefs=np.column_stack((modelcoefficients,np.array(names)))
      
      self.overall_result.coefficients=list(modelcoefficients)
      self.overall_result.feature_cols=names

      self.overall_result.tvalues=summary.tValues
      self.overall_result.pvalues=summary.pValues
      self.overall_result.coefficientsStandardErrors=summary.coefficientStandardErrors

      self.overall_result.nullDeviance=summary.nullDeviance
      self.overall_result.dispersion=summary.dispersion
      self.overall_result.residualDegreeOfFreedomNull=summary.residualDegreeOfFreedomNull
      self.overall_result.residualDegreeOfFreedom=summary.residualDegreeOfFreedom
      self.overall_result.deviance=summary.deviance
      self.overall_result.AIC=summary.aic



#%%
if __name__=='main':
    glr = RegressionModel_spark(
      filepath = '/home/ashleyubuntu/model_builder/car data.csv',
      label_col = "Selling_Price",
      selected_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Owner'],
      remove_columns = [],
      inferSchema = True,
      standardize = False,
      normalize = False,
      train_percent = 0.7,
      test_percent = 0.3,
      seed = 2022,
      family = "gaussian",
      link = "identity",
      max_iterations = 10,
      regularization_param = 0.3)
else:
  pass
    # glr = RegressionModel_spark(
    #   filepath = '/home/ashleyubuntu/model_builder/car data.csv',
    #   label_col = "Selling_Price",
    #   selected_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Owner'],
    #   remove_columns = [],
    #   inferSchema = True,
    #   standardize = False,
    #   normalize = False,
    #   train_percent = 0.7,
    #   test_percent = 0.3,
    #   seed = 2022,
    #   family = "gaussian",
    #   link = "identity",
    #   max_iterations = 10,
    #   regularization_param = 0.3)

    # pass

# %%
