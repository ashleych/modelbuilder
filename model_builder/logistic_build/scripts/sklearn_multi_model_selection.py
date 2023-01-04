# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
from pathlib import Path
from django.conf import settings
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector
# %%


class FeatureSelection():
    MAX_SHORT_LIST_FEATURES = 15

    def __init__(self, traindata_path,  label_col, feature_cols, short_list_max_features, max_features, min_features, cross_validation, exclude_features, fixed_features, variance_threshold, scoring, *args, **kwargs) -> None:
        # traindata=pd.read_csv(traindata_path)

        if os.path.exists(traindata_path):
            file_path = traindata_path
        else:
            file_path = os.path.join(
                Path(settings.BASE_DIR).parent, traindata_path)
            if not os.path.exists(file_path):
                ValueError("Input file doesnt exist")
        traindata = pd.read_csv(file_path)
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(traindata.drop(
            labels=[label_col], axis=1), traindata[label_col], test_size=0.8, stratify=traindata[label_col], random_state=41)
        del traindata
        self.max_features = max_features
        self.min_features = min_features
        self.feature_cols = feature_cols
        self.exclude_features = exclude_features
        self.fixed_features = fixed_features
        self.short_list_max_features = short_list_max_features
        self.cv = cross_validation
        self.models = None
        self.scoring = scoring
        self.remove_constant_columns()
        if variance_threshold:
            self.variance_threshold = variance_threshold
            self.remove_quasi_constant_columns(
                variance_threshold=variance_threshold)
        # self.remove_duplicate_columns()
        self.keep_numeric_columns_only()
        self.remove_correlated_columns()
        self.treat_nas()
        self.random_forest_feature_selector()
        self.exhaustive_model_builder()

    def remove_constant_columns(self):
        constant_filter = VarianceThreshold(threshold=0)
        constant_filter.fit(self.train_features)
        constant_columns = [
            column for column in self.train_features.columns if column not in self.train_features.columns[constant_filter.get_support()]]
        self.train_features.drop(labels=constant_columns, axis=1, inplace=True)

    def remove_quasi_constant_columns(self, variance_threshold=0.01):
        qconstant_filter = VarianceThreshold(threshold=variance_threshold)

        qconstant_filter.fit(self.train_features)

        qconstant_columns = [
            column for column in self.train_features.columns if column not in self.train_features.columns[qconstant_filter.get_support()]]
        self.train_features.drop(
            labels=qconstant_columns, axis=1, inplace=True)

    def keep_numeric_columns_only(self):
        num_colums = ['int16', 'int32', 'int64',
                      'float16', 'float32', 'float64']
        numerical_columns = list(
            self.train_features.select_dtypes(include=num_colums).columns)
        self.train_features = self.train_features[numerical_columns]

    def treat_nas(self):
        self.train_features.fillna(0, inplace=True)

    def remove_duplicate_columns(self):
        train_features_T = self.train_features.T
        train_features_T.shape
        print(train_features_T.duplicated().sum())
        unique_features = train_features_T.drop_duplicates(keep='first').T
        duplicated_features = [
            dup_col for dup_col in self.train_features.columns if dup_col not in unique_features.columns]
        duplicated_features

    def remove_correlated_columns(self):
        correlated_features = set()
        correlation_matrix = self.train_features.corr()

        # https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on?
        for i in range(len(correlation_matrix .columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > 0.8 and (correlation_matrix.columns[j] not in correlated_features):
                    colname = correlation_matrix.columns[i]
                    correlated_features.add(colname)

        self.train_features.drop(
            labels=correlated_features, inplace=True, axis=1)

    def random_forest_feature_selector(self):

        import math
        if self.short_list_max_features < self.max_features:
            self.short_list_features = self.max_features
        if self.short_list_max_features:
            n_estimators = min(self.short_list_max_features,
                               FeatureSelection.MAX_SHORT_LIST_FEATURES)
        else:
            n_estimators = int(math.sqrt(len(self.train_features.columns)))
        feature_selector = SequentialFeatureSelector(RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1), k_features=self.short_list_max_features, forward=True, verbose=2, scoring=self.scoring, cv=self.cv, n_jobs=1)
        self.short_list_features = feature_selector.fit(
            self.train_features, self.train_labels)
        # takes the best one where k=number of features, this may not be the best one as
        self.filtered_features_list = list(
            self.train_features.columns[list(self.short_list_features.k_feature_idx_)])
        # there may be models where the no of variables are lesser. See https://github.com/rasbt/mlxtend/issues/41 for discussion
        # For now, doesnt matter as this is merely being used to create a short list of features

    def exhaustive_model_builder(self):
        #  this will generate all combinations of variables in an exhaustive search, will take time for large datasets
        model_type = RandomForestClassifier(n_jobs=-1)
        feature_selector = ExhaustiveFeatureSelector(
            model_type, min_features=self.min_features, max_features=self.max_features, scoring=self.scoring, print_progress=True, cv=self.cv)
        self.models = feature_selector.fit(
            self.train_features[self.filtered_features_list], self.train_labels)
        # This contains results of all combinations of models
        self.models_all = self.models.subsets_


# %%
if __name__ == '__main__':
    santandar_data = pd.read_csv("/home/ashleyubuntu/model_builder/santander/santander_train_2k.csv")
    fp='/home/ashleyubuntu/model_builder/santander/santander_train_2k.csv'

    mfs = FeatureSelection(fp, 'TARGET', max_features=2,feature_cols=None,cross_validation=2,exclude_features=None,fixed_features=None,variance_threshold=0.01,scoring='roc_auc',
                           min_features=2, cv=4, short_list_max_features=4)
    print(mfs.models.subsets_)
    print(mfs.models)

# %%
