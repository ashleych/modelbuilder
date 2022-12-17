#%% 
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
import scorecardpy as sc

from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot as plt 

def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    return cls

# @auto_str

from dataclasses import dataclass

@auto_str
class ClassificationMetrics():
# https://www.kaggle.com/code/solaznog/mllib-spark-and-pyspark
    def __init__(self) -> None:
      self.auc=None
      self.coefficients=None

class OverallClassificationResults():
   def __init__(self) -> None:
    self.coefficients=None
    self.intercept=None
    self.train_result =None
    self.test_result =None


class LogisticRegressionModel_spark():
  def __init__(self, 
                seed: int = 2022,
                filepath = "/home/ashleyubuntu/model_builder/Merged_Dataset.csv",
                # filepath = "/home/ashleyubuntu/model_builder/model_builder/logistic_build/subset_merged.csv",
                label_col = 'def_trig',
                remove_cols_list = ["_c0", "Date", "Date.1"],
                prediction_cols = ["M97"],
                train_percent = 0.7,
                test_percent = 0.3,
                normalise=False
               ):
    
    self.spark  = self.get_spark_session()
    self.train_result = ClassificationMetrics() #initialise this 
    self.test_result = ClassificationMetrics() #initialise this 
    self.overall_result = OverallClassificationResults() #initialise this 
    self.df = self.read_csv_data(filepath)
    self.columns = self.df.columns
    import psutil

# Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    self.remove_cols(remove_cols_list)
    self.feature_cols = self.get_feature_colums(label_col)
    self.spark_output_colname=None
    self.stages = []
    self.add_assembler(label_col)
    if normalise:
      self.add_normaliser(label_col)
    self.train, self.test = self.train_test_split(train_percent, test_percent, seed)
    # self.
    self.build_pipeline()
    self.transform_data()
    self.lrModel= self.build_logistic_regression_model()
    self.get_coefficients_names()
    self.test_result.auc = self.plot_roc(self.piped_test_data,self.lrModel,type='test')
    self.train_result.auc = self.plot_roc(self.piped_test_data,self.lrModel,type='test')
    self.plot_roc(self.piped_train_data,self.lrModel,type='train')
    self.plot_precision_recall(type='train')
    self.plot_precision_recall(type='test')
    # self.create_results()
    # self.plot_beta_coeficients()
    # self.plot_area_under_ROC()
    # self.plot_precision_recall()
    # self.predict(prediction_cols)
    # self.get_area_under_ROC()
    # print("all ok")

  # def create_results(self):
  #   results=Results()
  #   results.coefficients =self.coefficients

  #   pass

  def get_spark_session(app_name:str = "Scorecard"):
      spark.sparkContext.stop()
      spark = SparkSession.builder.appName(app_name).config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").config('spark.executor.memory', '2g').config("spark.driver.host","localhost").getOrCreate()
      # http://localhost:4040/executors/ use this to check local spark session
      return spark    
    # def build_vector_assembler(self):
    #   vec_assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features_vector")
    #   self.stages+=
  def read_csv_data(self,filepath:str):
    if os.path.exists(filepath):
      df = self.spark.read.option("header",True).csv(filepath, inferSchema=True)
      print(df.columns)
    else:
      RuntimeError(f"Path {filepath} not found")
    return df

  def remove_cols(self, remove_cols_list):
      self.df = self.df.drop(*remove_cols_list)

  def get_feature_colums(self, label_col):
      cols = self.df.columns
      cols.remove(label_col) 
      return cols

  def get_correlations(self):

    correlations = self.df.sample(withReplacement=False, fraction=0.2, seed=44).select(self.columns[1:]).toPandas().corr()

    print('Correlations above 0.3: ', np.sum(np.sum(abs(correlations)>0.3)) - 201)

  def add_assembler(self,label_col : str):
    stages = []

    label_stringIdx = StringIndexer(inputCol = label_col, outputCol = 'label')
    self.stages += [label_stringIdx]
    numericCols = self.feature_cols
    assemblerInputs =  numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features_vector")
    self.stages += [assembler]
    self.spark_output_colname="features_vector"

    # return stages   
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
#     self.df = pipelineModel.transform(self.df)
#     selectedCols = ['label', 'features'] + self.feature_cols
#     self.df = self.df.select(selectedCols) 
# self.pipelineModel = santander_pipe.fit(training)
  def transform_data(self):
    self.piped_train_data = self.pipelineModel.transform(self.train)
    self.piped_test_data = self.pipelineModel.transform(self.test)

    # piped_train_data = self.pipelineModel.transform(self.train).select(["ID_code", "label", "features"])
    # piped_test_data = self.pipelineModel.transform(self.test).select(["ID_code", "label", "features"])


  def build_logistic_regression_model(self):
    lr = LogisticRegression(featuresCol = self.spark_output_colname, labelCol = 'label')
    lrModel = lr.fit(self.piped_train_data)

    return lrModel

  def get_coefficients_names(self):
    # Source : https://turingintern2018.github.io/sparkairplane.html
      import numpy as np
      modelcoefficients=np.array(self.lrModel.coefficients)
      names_with_idx=[]
      if self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('numeric'):
          names_with_idx += self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('numeric')
      
      if self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('binary'):
          names_with_idx+= self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('binary')
      names=[x['name'] for x in sorted(names_with_idx, key= lambda x:x["idx"])]
      matchcoefs=np.column_stack((modelcoefficients,np.array(names)))
      self.overall_result.coefficients=matchcoefs
      self.overall_result.feature_names=np.array(names)
      # import pandas as pd

      # matchcoefsdf=pd.DataFrame(matchcoefs)

      # matchcoefsdf.columns=['Coefvalue', 'Feature']

      # print(matchcoefsdf)
      # self.results.coefficients=matchcoefsdf
      # return matchcoefsdf['Coefvalue']

  def plot_roc(self,data, model,type="test"):
    pred_ = model.transform(data)
    self.pred_=pred_
    pred_pd_ = pred_.select(['label', 'prediction', 'probability']).toPandas()

    pred_pd_['probability'] = pred_pd_['probability'].map(lambda x: list(x))
    pred_pd_['encoded_label'] = pred_pd_['label'].map(lambda x: np.eye(2)[int(x)])

    y_pred_ = np.array(pred_pd_['probability'].tolist())
    y_true_ = np.array(pred_pd_['encoded_label'].tolist())

    fpr_, tpr_, threshold_ = roc_curve(y_score=y_pred_[:,0], y_true=y_true_[:,0])
    auc_ = auc(fpr_, tpr_)

    if type=='test':
      self.test_result.precision, self.test_result.recall, self.test_result.thresholds = precision_recall_curve(y_true_[:,0], y_pred_[:,0])

    return auc_
    # setattr(self.result,"auc_"+type,auc_)
    # plt.figure()
    # plt.plot([0,1], [0,1], '--', color='orange')
    # plt.plot(fpr_, tpr_, label='auc = {:.3f}'.format(auc_))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title(f'ROC curve - {type} data')
    # plt.legend(loc='lower right')
    # plt.grid()
    # setattr(self.result,"plot_roc_"+type,plt)
    # plt.show()

  def plot_beta_coeficients(self):
    beta = np.sort(self.lrModel.coefficients)
    plt.plot(beta)
    plt.ylabel('Beta Coefficients')
    # plt.show()

  def plot_area_under_ROC(self):
    self.trainingSummary = self.lrModel.summary
    self.roc = self.trainingSummary.roc.toPandas()
    # plt.plot(self.roc['FPR'],self.roc['TPR'])
    # plt.ylabel('False Positive Rate')
    # plt.xlabel('True Positive Rate')
    # plt.title('ROC Curve')
    # # plt.show()
    # print('Training set areaUnderROC: ' + str(self.trainingSummary.areaUnderROC))   

  def plot_precision_recall(self,type='test'):
#     from pyspark.mllib.evaluation import BinaryClassificationMetrics
#     predictionAndLabels = self.piped_test_data.map(lambda lp: (float(self.lrModel.predict(lp[self.spark_output_colname])), lp.label))
#     self.lrModel.summary.pr.toPandas()
#     metrics = BinaryClassificationMetrics(predictionAndLabels)

# # Area under precision-recall curve
#     print("Area under PR = %s" % metrics.areaUnderPR)
    from pyspark.mllib.evaluation import BinaryClassificationMetrics

#     # Area under ROC curve
#     print("Area under ROC = %s" % metrics.areaUnderROC)
    if type=='train':
      pr_df =self.lrModel.summary.pr.toPandas()
      self.train_result.precision =pr_df['precision'].to_list() 
      self.train_result.recall = pr_df['recall'].to_list()
      # plt.plot(self.result.pr_train['recall'],self.result.pr_train['precision'])
      # plt.ylabel('Precision')
      # plt.xlabel('Recall')
    else:
      pass
      # predictions= self.lrModel.transform(self.piped_test_data)  
      # results = predictions.select(['prediction', 'label'])
      # # predictionAndLabels=results.rdd
      # evaluator = BinaryClassificationEvaluator()
      # self.test_result.areaUnderPR = evaluator.evaluate(predictions,{evaluator.metricName: "areaUnderPR"})

      # pred_ = self.lrModel.transform(self.piped_test_data)
      # # self.pred_=pred_
      # pred_pd_ = pred_.select(['label', 'prediction', 'probability']).toPandas()

      # pred_pd_['probability'] = pred_pd_['probability'].map(lambda x: list(x))
      # pred_pd_['encoded_label'] = pred_pd_['label'].map(lambda x: np.eye(2)[int(x)])

      # y_pred_ = np.array(pred_pd_['probability'].tolist())
      # y_true_ = np.array(pred_pd_['encoded_label'].tolist())

      # self.test_result.precision, self.test_result.recall, self.test_result.thresholds = precision_recall_curve(y_true_[:,0], y_pred_[:,0])
      # metrics = BinaryClassificationMetrics(predictionAndLabels)

      # Area under precision-recall curve
      # print("Area under PR = %s" % metrics.areaUnderPR)

      # # Area under ROC curve
      # print("Area under ROC = %s" % metrics.areaUnderROC)
    # plt.show() 

  def predict(self, data):
    self.result.predictions = self.lrModel.transform(data)  
    # self.predictions.select(prediction_cols).show(10)

  def get_area_under_ROC(self):
    evaluator = BinaryClassificationEvaluator()
    self.area_under_roc = evaluator.evaluate(self.result.predictions)
    # print('Test Area Under ROC', evaluator.evaluate(self.predictions)) 

#%%
if __name__=='main':
    lr_test = LogisticRegressionModel_spark()    
    filepath = "/home/ashleyubuntu/model_builder/Merged_Dataset.csv"
else:
    lr_test = LogisticRegressionModel_spark()    
    print(str(lr_test.test_result))

# %%
# import h2o
# # h2o.init()
# # h2o.init(max_mem_size = "8g")
# h2o.init(max_mem_size = "8g")
# # h2o.demo("glm")
# # %%
# filepath='/home/ashleyubuntu/model_builder/output/new_exp_2_Exp_id_135.csv'
# training_data = h2o.import_file(filepath)
# #%%
# # Set the predictors and response:
# predictors = ["abs.C1.", "abs.C2.", "abs.C3.", "abs.C4.", "abs.C5.""]
# response = "def_trig"

# # Build and train the model:
# model = H2OGeneralizedLinearEstimator(family="binomial",
#                                       lambda_=0,
#                                       compute_p_values=True,
#                                       dispersion_parameter_method="ml",
#                                       init_dispersion_parameter=1.1,
#                                       dispersion_epsilon=1e-4,
#                                       max_iterations_dispersion=100)
                                      
# model.train(x=predictors, y=response, training_frame=training_data)

# # Retrieve the estimated dispersion:
# model._model_json["output"]["dispersion"]

#%%
def evaluation_metrics_mllib(model,training,test):

  # bring more from https://github.com/ShichenXie/scorecardpy/blob/master/scorecardpy/perf.py
        from pyspark.mllib.classification import LogisticRegressionWithLBFGS
        from pyspark.mllib.evaluation import BinaryClassificationMetrics
        from pyspark.mllib.util import MLUtils

        # Several of the methods available in scala are currently missing from pyspark
        # Load training data in LIBSVM format
        # data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_binary_classification_data.txt")

        # Split data into training (60%) and test (40%)
        # training, test = data.randomSplit([0.6, 0.4], seed=11L)
        training.cache()

        # Run training algorithm to build the model
        # model = LogisticRegressionWithLBFGS.train(training)

        # Compute raw scores on the test set
        predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))

        # Instantiate metrics object
        metrics = BinaryClassificationMetrics(predictionAndLabels)

        # Area under precision-recall curve
        print("Area under PR = %s" % metrics.areaUnderPR)

        # Area under ROC curve
        print("Area under ROC = %s" % metrics.areaUnderROC)


#%%
# from pysparkling import *
# import h2o
# hc = H2OContext.getOrCreate()
# %%

# %%
# import pandas as pd
# filepath = "/home/ashleyubuntu/model_builder/Merged_Dataset.csv"
# pd.read_csv(filepath,nrows=50000).to_csv('subset_merged.csv')

# %%