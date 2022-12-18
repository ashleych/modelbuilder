#%%
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
from pyspark.sql.types import StructType
from pyspark.sql.types import IntegerType,BooleanType,DateType,StringType
# %%
filepath ='/home/ashleyubuntu/model_builder/model_builder/logistic_build/subset_merged.csv'

spark = SparkSession.builder.appName('app_namassde').config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").config('spark.executor.memory', '2g').config("spark.driver.host","localhost").getOrCreate()
df = spark.read.option("header",True).csv(filepath, inferSchema=True)

# %%
df_schema=df.schema
# %%
type(df_schema)
struct1 = StructType().add("f1", StringType(), True).add("f2", StringType(), True, None)

# %%
    # import org.apache.spark.sql.types.{DataType, StructType}
# val newSchema = DataType.fromJson(jsonString).asInstanceOf[StructType]

jsonSchema=df_schema.jsonValue()
# %%
new_schema = StructType.fromJson(jsonSchema)
new_schema
# %%
