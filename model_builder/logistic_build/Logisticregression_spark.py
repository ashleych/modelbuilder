# %%
import pandas as pd
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
import plotly.express as px
from pathlib import Path

from django.conf import settings
from .logistic_results import OverallClassificationResults, auto_str, ClassificationMetrics, plot_precision_recall, plot_roc


class LogisticRegressionModel_spark():
    def __init__(self,
                 seed: int = 2022,
                 # filepath = "/home/ashleyubuntu/model_builder/csv",
                 filepath="/home/ashleyubuntu/model_builder/model_builder/logistic_build/subset_merged.csv",
                 label_col='def_trig',
                 remove_cols_list=["_c0", "Date", "Date.1"],
                 prediction_cols=["M97"],
                 train_percent=0.7,
                 test_percent=0.3,
                 normalise=False,
                 inferSchema=True
                 ):

        self.spark = self.get_spark_session()
        self.train_result = ClassificationMetrics(type='train')  # initialise this
        self.test_result = ClassificationMetrics(type='test')  # initialise this
        self.overall_result = OverallClassificationResults()  # initialise this
        self.df = self.read_csv_data(filepath, inferSchema=inferSchema)
        self.columns = self.df.columns
        # import psutil

# Getting % usage of virtual_memory ( 3rd field)
        # print('RAM memory % used:', psutil.virtual_memory()[2])
        # # Getting usage of virtual_memory in GB ( 4th field)
        # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
        self.remove_cols(remove_cols_list)
        self.feature_cols = self.get_feature_colums(label_col)
        self.spark_output_colname = None
        self.stages = []
        self.add_assembler(label_col)
        if normalise:
            self.add_normaliser(label_col)
        self.train, self.test = self.train_test_split_spark(train_percent, test_percent, seed)
        self.overall_result.train_nrows = str(self.train.count())
        self.overall_result.test_nrows = str(self.test.count())
        print("self.train.count " + str(self.train.count()))
        print("self.test.count " + str(self.test.count()))
        print("self.df.count " + str(self.df.count()))
        # self.
        self.build_pipeline()
        self.transform_data()
        self.lrModel = self.build_logistic_regression_model()
        self.get_coefficients_names()

        self.plot_roc_and_pr(self.piped_test_data, self.lrModel, type='test')
        self.plot_roc_and_pr(self.piped_train_data, self.lrModel, type='train')
        self.plot_precision_recall(type='train')
        self.plot_precision_recall(type='test')

    def get_spark_session(app_name: str = "Scorecard"):

        spark = SparkSession.builder.appName(app_name).config("spark.memory.offHeap.enabled", "true").config("spark.memory.offHeap.size",
                                                                                                             "10g").config('spark.executor.memory', '2g').config("spark.driver.host", "localhost").getOrCreate()
        # http://localhost:4040/executors/ use this to check local spark session
        return spark
        # def build_vector_assembler(self):
        #   vec_assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features_vector")
        #   self.stages+=

    def read_csv_data(self, filepath: str, inferSchema):
        if os.path.exists(filepath):
            df = self.spark.read.option("header", True).csv(filepath, inferSchema=inferSchema)
            # print(df.columns)
        else:
            abs_file_path = os.path.join(Path(settings.BASE_DIR).parent, filepath)
            print(abs_file_path)
            if os.path.exists(abs_file_path):
                df = self.spark.read.option("header", True).csv(abs_file_path, inferSchema=inferSchema)
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

        print('Correlations above 0.3: ', np.sum(np.sum(abs(correlations) > 0.3)) - 201)

    def add_assembler(self, label_col: str):
        stages = []

        label_stringIdx = StringIndexer(inputCol=label_col, outputCol='label')
        self.stages += [label_stringIdx]
        numericCols = self.feature_cols
        assemblerInputs = numericCols
        assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features_vector")
        self.stages += [assembler]
        self.spark_output_colname = "features_vector"

        # return stages
    def add_normaliser(self):
        from pyspark.ml.feature import StandardScaler

        standardScaler = StandardScaler()
        standardScaler.setWithMean(True)
        standardScaler.setWithStd(True)
        standardScaler.setInputCol(self.spark_output_colname)
        standardScaler.setOutputCol("features")
        self.spark_output_colname = "features"
        self.stages += [standardScaler]

    def train_test_split_spark(self, train_percent, test_percent, seed):
        train, test = self.df.randomSplit([0.8, 0.2], seed=seed)
        print("Train -test split in spark")
        print(train_percent)
        print(test_percent)
        print(self.df.count())
        print(train.count())
        return train, test

    def build_pipeline(self):
        pipeline = Pipeline(stages=self.stages)
        self.pipelineModel = pipeline.fit(self.train)

    def transform_data(self):
        self.piped_train_data = self.pipelineModel.transform(self.train)
        self.piped_test_data = self.pipelineModel.transform(self.test)

    def build_logistic_regression_model(self):
        lr = LogisticRegression(featuresCol=self.spark_output_colname, labelCol='label')
        lrModel = lr.fit(self.piped_train_data)

        return lrModel

    def get_coefficients_names(self):
        # Source : https://turingintern2018.github.io/sparkairplane.html
        import numpy as np
        modelcoefficients = np.array(self.lrModel.coefficients)
        names_with_idx = []
        if self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('numeric'):
            names_with_idx += self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('numeric')

        if self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('binary'):
            names_with_idx += self.piped_train_data.schema[self.spark_output_colname].metadata["ml_attr"]["attrs"].get('binary')
        names = [x['name'] for x in sorted(names_with_idx, key=lambda x:x["idx"])]
        matchcoefs = np.column_stack((modelcoefficients, np.array(names)))
        self.overall_result.coefficients = list(modelcoefficients)
        self.overall_result.features = names

    def plot_roc_and_pr(self, data, model, type="test"):
        # Please note that this also computes the precision recall curve
        #  this is being done so as to avoid a recompute
        if type == 'train':
            self.trainingSummary = self.lrModel.summary
            self.roc = self.trainingSummary.roc.toPandas()
            self.train_result.FPR = self.roc['FPR']
            self.train_result.TPR = self.roc['TPR']
            self.train_result.areaUnderROC = self.trainingSummary.areaUnderROC
        else:
            pred_ = model.transform(data)
            self.pred_ = pred_
            pred_pd_ = pred_.select(['label', 'prediction', 'probability']).toPandas()

            pred_pd_['probability'] = pred_pd_['probability'].map(lambda x: list(x))
            pred_pd_['encoded_label'] = pred_pd_['label'].map(lambda x: np.eye(2)[int(x)])

            y_pred_ = np.array(pred_pd_['probability'].tolist())
            y_true_ = np.array(pred_pd_['encoded_label'].tolist())

            fpr_, tpr_, threshold_ = roc_curve(y_score=y_pred_[:, 0], y_true=y_true_[:, 0])
            auc_ = auc(fpr_, tpr_)
            self.test_result.FPR = fpr_
            self.test_result.TPR = tpr_
            self.test_result.areaUnderROC = auc_
        if type == 'test':
            self.test_result.precision_plot_data, self.test_result.recall_plot_data, _ = precision_recall_curve(y_true_[:, 0], y_pred_[:, 0])

        return

    def plot_beta_coeficients(self):
        beta = np.sort(self.lrModel.coefficients)
        plt.plot(beta)
        plt.ylabel('Beta Coefficients')
        # plt.show()

    def plot_area_under_ROC(self):
        pass

        # print('Training set areaUnderROC: ' + str(self.trainingSummary.areaUnderROC))

    def plot_precision_recall(self, type='test'):

        # # Area under precision-recall curve
        #     print("Area under PR = %s" % metrics.areaUnderPR)
        from pyspark.mllib.evaluation import BinaryClassificationMetrics

#     # Area under ROC curve
#     print("Area under ROC = %s" % metrics.areaUnderROC)
        if type == 'train':
            pr_df = self.lrModel.summary.pr.toPandas()
            # self.train_result.areaUnderPR = self.lrModel.summary.areaUnderPR

            self.train_result.precision_plot_data = pr_df['precision'].to_list()
            self.train_result.recall_plot_data = pr_df['recall'].to_list()

            predictions = self.lrModel.transform(self.piped_train_data)
            evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR')
            self.train_result.areaUnderPR = evaluator.evaluate(predictions)
        else:
            predictions = self.lrModel.transform(self.piped_test_data)
            evaluator = BinaryClassificationEvaluator(metricName='areaUnderPR')
            # good example is in the link below
            #  https://towardsdatascience.com/predict-customer-churn-using-pyspark-machine-learning-519e866449b5
            self.test_result.areaUnderPR = evaluator.evaluate(predictions)

    def predict(self, data):
        self.result.predictions = self.lrModel.transform(data)
        # self.predictions.select(prediction_cols).show(10)/

    def get_area_under_ROC(self):
        evaluator = BinaryClassificationEvaluator()
        self.area_under_roc = evaluator.evaluate(self.result.predictions)
        # print('Test Area Under ROC', evaluator.evaluate(self.predictions))


# %%
