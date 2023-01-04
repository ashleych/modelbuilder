# %%
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from django.conf import settings
import os
import json
from typing import List
# import scorecardpy as sc

import statsmodels.formula.api as smf
# from sklearn.metrics import auc, roc_curve, accuracy_score, roc_auc_score
from matplotlib import pyplot as plt
from .logistic_results import OverallClassificationResults, auto_str, ClassificationMetrics, plot_precision_recall, plot_roc
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, average_precision_score, auc, roc_curve
from sklearn.metrics import PrecisionRecallDisplay

class LogisticRegressionModel_sklearn():
    def __init__(self,
                 seed: int = 2022,
                 # filepath = "/home/ashleyubuntu/model_builder/Merged_Dataset.csv",
                 filepath="/home/ashleyubuntu/model_builder/model_builder/logistic_build/subset_merged.csv",
                 label_col='def_trig',
                 exclude_features=None,
                 prediction_cols=["M97"],
                 train_percent=0.7,
                 test_percent=0.3,
                 normalise=False,
                 inferSchema=True,
                 stratify=True,
                 logistic_threshold=0.5
                 ):

        # self.spark  = self.get_spark_session()
        self.train_result = ClassificationMetrics(type='train')  # initialise
        self.test_result = ClassificationMetrics(type='test')  # initialise
        self.overall_result = OverallClassificationResults()  # initialise this
        self.input_data = self.read_csv_data(filepath, inferSchema=inferSchema)
        self.label_col = label_col
        self.logistic_threshold = logistic_threshold
        # self.features = self.input_data.columns
        self.exclude_features = exclude_features
        self.train_size = train_percent
        self.test_size = test_percent
        self.stratify = stratify

        self.remove_cols()
        if normalise:
            self.add_normaliser(label_col)
        self.train_split()
        # print("self.train.count " + str(self.train.count()))
        # print("self.test.count " + str(self.test.count()))
        # self.
        self.build_pipeline()
        self.transform_data()
        # self.lrModel = self.build_logistic_regression_model()
        self.lrModel = self.build_logistic_regression_model_using_sklearn()
        self.aggregate_test_accuracy_results(type='train')
        self.aggregate_test_accuracy_results(type='test')
        self.get_coefficients_names()

    def read_csv_data(self, filepath: str, inferSchema):
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
        else:
            abs_file_path = os.path.join(
                Path(settings.BASE_DIR).parent, filepath)
            print(abs_file_path)
            if os.path.exists(abs_file_path):
                df = pd.read_csv(abs_file_path)
            else:
                RuntimeError(f"Path {filepath} not found")
        return df

    def remove_cols(self,):
        if self.exclude_features:
            self.input_data = self.input_data.drop(*self.exclude_features)

    def get_feature_colums(self, label_col):
        cols = self.input_data.columns
        cols.remove(label_col)
        return cols

    def train_split(self):
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(self.input_data.drop(
            columns=self.label_col), self.input_data[self.label_col], train_size=self.train_size, test_size=self.test_size, stratify=self.input_data[self.label_col], random_state=1)
        self.input_data = None
        self.overall_result.train_nrows=self.train_features.shape[0]
        self.overall_result.test_nrows=self.test_features.shape[0]

    def build_pipeline(self):
        pass

    def transform_data(self):
        pass

    def build_logistic_regression_model_using_sklearn(self):
        from sklearn.linear_model import LogisticRegression
        log_reg = LogisticRegression(random_state=123)

        log_reg.fit(self.train_features, self.train_labels)
        self.Y_preds_test = log_reg.predict(self.test_features)
        self.Y_preds_train = log_reg.predict(self.train_features)
        self.Y_preds_prob_test = log_reg.predict_proba(self.test_features)
        self.Y_preds_prob_train = log_reg.predict_proba(self.train_features)
        self.logit = log_reg
        # self.Y_preds = Y_preds
        return

    def build_logistic_regression_using_statsmodels(self):
        self.train_features[label_col] = self.train_labels

        # logit = LogisticRegression(solver='lbfgs')
        # logit.fit(x_train,y_train)0:
        # logit = smf.logit("TARGET ~ ind_var12_0 + saldo_medio_var5_hace2", data=self.train_features).fit()
        all_features = [
            column for column in self.train_features.columns if column != self.label_col]
        formula_all_columns = "+".join(all_features)
        logit = smf.logit(f"TARGET ~  {formula_all_columns}", data=self.train_features).fit(
            method='bfgs', maxiter=1000)
        self.logit = logit
        self.Y_preds_prob_test = logit.predict(self.test_features)
        self.Y_preds_prob_train = logit.predict(self.train_features)

        self.Y_preds_train = self.Y_preds_prob_train.map(
            lambda x: 1 if x > self.logistic_threshold else 0)
        self.Y_preds_test = self.Y_preds_prob_test.map(
            lambda x: 1 if x > self.logistic_threshold else 0)

    @staticmethod
    def binary_classification_performance(y_ground_truth, y_pred, y_pred_prob, type):
        result = ClassificationMetrics(type=type)
        result.tp, result.fp, result.fn, result.tn = confusion_matrix(
            y_ground_truth, y_pred).ravel()
        result.accuracy = round(accuracy_score(
            y_pred=y_pred, y_true=y_ground_truth), 2)

        result.precision = round(precision_score(
            y_pred=y_pred, y_true=y_ground_truth), 2)
        result.recall = round(recall_score(
            y_pred=y_pred, y_true=y_ground_truth), 2)
        result.f1_score = round(
            2 * result.precision * result.recall / (result.precision + result.recall), 2)
        result.specificity = round(result.tn / (result.tn + result.fp), 2)
        result.npv = round(result.tn / (result.tn + result.fn), 2)
        # average_precision_score(y_test, y_score)
        # except:
        #     print("Unable to generate Precision metrics,perhaps because no predictions as 1")
        # here we need to pass the probability of the positive class
        result.areaUnderROC = round(roc_auc_score(
            y_score=y_pred_prob, y_true=y_ground_truth), 2)

        result.FPR, result.TPR, _ = roc_curve(
            y_ground_truth, y_pred_prob)
        # from sklearn.datasets import make_classification
        # X, y = make_classification(n_samples=500, random_state=0)
        # from sklearn.linear_model import LogisticRegression
        # model = LogisticRegression()
        # model.fit(X, y)
        # y_score = model.predict_proba(X)[:, 1]

        # result.precision_plot_data, result.recall_plot_data, _ = precision_recall_curve(y, y_score)

        result.precision_plot_data, result.recall_plot_data, _ = precision_recall_curve(y_ground_truth, y_pred_prob, pos_label=1)
        # df=pd.DataFrame({'precision':result.precision_plot_data, 'recall': result.recall_plot_data})
        # df.to_csv('precision_recall.csv')
        # PrecisionRecallDisplay.from_predictions(y_ground_truth, y_pred_prob)
        # plt.show()
        # fig.show()
        # result.average_prevision_score = average_precision_score(y_ground_truth, y_pred_prob)
        result.areaUnderPR = round(
            auc(result.recall_plot_data, result.precision_plot_data), 2)
        fig = plot_precision_recall(result.recall_plot_data, result.precision_plot_data, result.areaUnderPR)
        return result

    def aggregate_test_accuracy_results(self, type):
        # https://towardsdatascience.com/calculating-and-setting-thresholds-to-optimise-logistic-regression-performance-c77e6d112d7e#:~:text=The%20logistic%20regression%20assigns%20each,0.5%20is%20the%20default%20threshold.
        pred_labels = getattr(self, 'Y_preds_' + type)
        pred_scores = getattr(self, 'Y_preds_prob_' + type)[:, 1]
        ground_truth_labels = getattr(self, type + "_labels")

        result = LogisticRegressionModel_sklearn.binary_classification_performance(
            ground_truth_labels, pred_labels, pred_scores, type=type)

        setattr(self, type + '_result', result)

    def get_coefficients_names(self):
        # Source : https://turingintern2018.github.io/sparkairplane.html
        self.overall_result.coefficients = json.dumps(
            list(self.logit.coef_[0, :]))
        self.overall_result.intercept = self.logit.intercept_[0]
        self.overall_result.features = json.dumps(
            list(self.train_features.columns))

    def plot_beta_coeficients(self):
        beta = np.sort(self.lrModel.coefficients)
        plt.plot(beta)
        plt.ylabel('Beta Coefficients')
        # plt.show()

    def predict(self, data):
        self.result.predictions = self.lrModel.transform(data)
        # self.predictions.select(prediction_cols).show(10)/
# %%
# from sklearn.datasets import make_classification
# import plotly.express as px
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_recall_curve, auc
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=500, random_state=0)
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_recall_curve
# model = LogisticRegression()
# model.fit(X, y)
# y_score = model.predict_proba(X)[:, 1]

# precision,recall,t = precision_recall_curve(y, y_score)
# fig = px.area(
#     x=recall, y=precision,
#     title=f'Precision-Recall Curve ',
#     labels=dict(x='Recall', y='Precision'),
#     width=700, height=500
# )
# fig.add_shape(
#     type='line', line=dict(dash='dash'),
#     x0=0, x1=1, y0=1, y1=0
# )
# fig.update_yaxes(scaleanchor="x", scaleratio=1)
# fig.update_xaxes(constrain='domain')

# fig.show()
#%%

if __name__ == '__main__':
    # if True:
    fp = '/home/ashleyubuntu/model_builder/santander/santander_train_2k.csv'
    fp = '/home/ashleyubuntu/model_builder/santander/santander_train_2k.csv'
    fixed_features = ['num_med_var45_ult3', 'saldo_medio_var5_hace3']
    fixed_features = None
    label_col = 'TARGET'
    lr_test = LogisticRegressionModel_sklearn(filepath=fp, label_col=label_col)
    filepath = "/home/ashleyubuntu/model_builder/Merged_Dataset.csv"
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.metrics import PrecisionRecallDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    # X, y = make_classification(random_state=1)
    import pandas as pd

    fp = '/home/ashleyubuntu/model_builder/sample_classification.csv'
    df = pd.read_csv(fp)
    X = df.drop(columns='TARGET')
    y = df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=1)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    PrecisionRecallDisplay.from_predictions(
        y_test, y_pred)

    plt.show()
else:
    pass
    # fp = '/home/ashleyubuntu/model_builder/santander/santander_train_2k.csv'
    # fixed_features = ['num_med_var45_ult3', 'saldo_medio_var5_hace3']
    # label_col = 'TARGET'
    # # lr_test = LogisticRegressionModel_sklearn(filepath=fp, label_col=label_col)
    # filepath = "/home/ashleyubuntu/model_builder/Merged_Dataset.csv"
    # import matplotlib.pyplot as plt
    # from sklearn.datasets import make_classification
    # from sklearn.metrics import PrecisionRecallDisplay
    # from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LogisticRegression
    # X, y = make_classification(random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, random_state=0)
    # clf = LogisticRegression()
    # clf.fit(X_train, y_train)

    # y_pred = clf.predict_proba(X_test)[:, 1]
    # PrecisionRecallDisplay.from_predictions(
    #     y_test, y_pred)

    # plt.show()

# %%

# %%
