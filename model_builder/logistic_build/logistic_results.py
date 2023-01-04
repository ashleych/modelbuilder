from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot as plt 
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go

def plot_roc(fpr,tpr,auc):
  fig = px.area(
      x=fpr, y=tpr,
      title=f'ROC Curve (AUC={auc:.4f})',
      labels=dict(x='False Positive Rate', y='True Positive Rate'),
      width=700, height=500
  )
  fig.add_shape(
      type='line', line=dict(dash='dash'),
      x0=0, x1=1, y0=0, y1=1
  )

  fig.update_yaxes(scaleanchor="x", scaleratio=1)
  fig.update_xaxes(constrain='domain')
  # fig.show()
  return fig


def plot_precision_recall(recall_plot_data, precision_plot_data, areaUnderPR):

        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )


        # fig.add_trace(go.Scatter(x = recall_plot_data, y = precision_plot_data,
        #                 name = "Precision" + str(areaUnderPR),
        #                 line = dict(color = ('lightcoral'), width = 2), fill='tozeroy'))

        # fig.update_layout(
        #     xaxis_title='Recall-new',
        #     yaxis_title='Precision',
        #     yaxis=dict(scaleanchor="x", scaleratio=1),
        #     xaxis=dict(constrain='domain'),
        #     width=700, height=500
        # )
        fig = px.area(
            x=recall_plot_data, y=precision_plot_data,
            title=f'Precision-Recall Curve (AUC=1)',
            labels=dict(x='Recall', y='Precision'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=1, y1=0
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        
        # fig.show()
    # fig = px.scatter(
    # x=recall_plot_data, y=precision_plot_data,mode='lines',
    # title=f'precision_plot_data-recall_plot_data Curve (Area under PR={areaUnderPR:.4f})',
    # labels=dict(x='recall_plot_data', y='precision_plot_data'),
    # width=700, height=500)
    # fig.add_shape(type='line', line=dict(dash='dash'),x0=0, x1=1, y0=1, y1=0)
    # fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # fig.update_xaxes(constrain='domain')
        return fig

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
import json
@auto_str
class ClassificationMetrics():
# https://www.kaggle.com/code/solaznog/mllib-spark-and-pyspark
    def __init__(self, type) -> None:
        self._FPR=None
        self._TPR=None
        self._precision_plot_data=None
        self._recall_plot_data=None
        self.type=type
      
    @property
    def FPR(self):
        return json.loads(self._FPR)
      
    # a setter function
    @FPR.setter
    def FPR(self, a):
        self._FPR = json.dumps(list(a))

    @property
    def TPR(self):
        return json.loads(self._TPR)
      
    # a setter function
    @TPR.setter
    def TPR(self, a):
        self._TPR = json.dumps(list(a))

    @property
    def precision_plot_data(self):
        return json.loads(self._precision_plot_data)
      
    # a setter function
    @precision_plot_data.setter
    def precision_plot_data(self, a):
        self._precision_plot_data = json.dumps(list(a))
    @property
    def recall_plot_data(self):
        return json.loads(self._recall_plot_data)
      
    # a setter function
    @recall_plot_data.setter
    def recall_plot_data(self, a):
        self._recall_plot_data = json.dumps(list(a))
      
    @property
    def all_attributes(self):
        all_attrib=dict(vars(self), FPR=self.FPR,TPR=self.TPR,precision_plot_data=self.precision_plot_data,recall_plot_data=self.recall_plot_data)
        keys= list(all_attrib.keys()) 
        for key in keys:
          if key.startswith("_"): #remove all private keys
            all_attrib.pop(key, 'No Key found')

        return all_attrib

class OverallClassificationResults():
   def __init__(self) -> None:
    self.coefficients=None
    self.features=None
    self.intercept=None
    self.train_result =None
    self.test_result =None

