#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score

santandar_data = pd.read_csv("./santander/train.csv", nrows=40000)
santandar_data.shape
#%%

class FeatureSelection():
    MAX_SHORT_LIST_FEATURES=15
    def __init__(self,input_data,label_col,short_list_max_features,max_features,min_features,cv) -> None:
        
        self.train_features, self.test_features, self.train_labels, self.test_labels=train_test_split(
            input_data.drop(labels=[label_col], axis=1),
            input_data[label_col],
            test_size=0.8,stratify=input_data[label_col],
            random_state=41)
        self.max_features=max_features
        self.min_features=min_features
        self.short_list_max_features=short_list_max_features
        self.cv=cv
        self.remove_constant_columns()
        self.remove_quasi_constant_columns(variance_threshold=0.01)
        # self.remove_duplicate_columns()
        self.keep_numeric_columns_only()
        self.remove_correlated_columns()
        self.random_forest_feature_selector()
        self.exhaustive_model_builder()
    
    def remove_constant_columns(self):
        constant_filter = VarianceThreshold(threshold=0)
        constant_filter.fit(self.train_features)
        constant_columns = [column for column in self.train_features.columns if column not in self.train_features.columns[constant_filter.get_support()]]
        self.train_features.drop(labels=constant_columns, axis=1, inplace=True)

    def remove_quasi_constant_columns(self,variance_threshold=0.01):
        qconstant_filter = VarianceThreshold(threshold=variance_threshold)

        qconstant_filter.fit(self.train_features)

        qconstant_columns = [column for column in self.train_features.columns if column not in self.train_features.columns[qconstant_filter.get_support()]]
        self.train_features.drop(labels=qconstant_columns, axis=1, inplace=True)


    def keep_numeric_columns_only(self):
        num_colums = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numerical_columns = list(self.train_features.select_dtypes(include=num_colums).columns)
        self.train_features = self.train_features[numerical_columns]
    
    def remove_duplicate_columns(self):
        train_features_T = self.train_features.T
        train_features_T.shape
        print(train_features_T.duplicated().sum())
        unique_features = train_features_T.drop_duplicates(keep='first').T
        duplicated_features = [dup_col for dup_col in self.train_features.columns if dup_col not in unique_features.columns]
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
                
        self.train_features.drop(labels=correlated_features,inplace=True,axis=1)

    def random_forest_feature_selector(self):
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from mlxtend.feature_selection import SequentialFeatureSelector
        import math
        if self.short_list_max_features:
            n_estimators= min(self.short_list_max_features,FeatureSelection.MAX_SHORT_LIST_FEATURES)
        else:
            n_estimators=int(math.sqrt(len(self.train_features.columns)))
        feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_estimators=n_estimators,n_jobs=-1), k_features=self.short_list_max_features, forward=True, verbose=2, scoring='roc_auc', cv=self.cv,n_jobs=1)
        self.features = feature_selector.fit(np.array(self.train_features.fillna(0)), self.train_labels)
        self.filtered_features=list(self.train_features.columns[list(self.features.k_feature_idx_)])
        import json
        js=json.dumps(list(self.filtered_features))
        with open("sample.json", "w") as outfile:
            outfile.write(js)
    
    def exhaustive_model_builder(self):
        model_type = RandomForestClassifier(n_jobs=-1)
        feature_selector = ExhaustiveFeatureSelector(model_type, min_features=self.min_features, max_features=self.max_features, scoring='roc_auc', print_progress=True, cv=self.cv)
        self.models = feature_selector.fit(np.array(self.train_features[self.filtered_features].fillna(0)),self.train_labels)

#%%
santandar_data = pd.read_csv("./santander/train.csv", nrows=30000)
mfs=FeatureSelection(santandar_data,'TARGET',max_features=2,min_features=2,cv=0,short_list_max_features=2)
print(mfs.subsets_)
print(mfs.models)
# %%

# #%%
# # Drop duplicates

# # Next, we need to simply apply this filter to our training set as shown in the following example:

# constant_filter.fit(train_features)
# # Now to get all the features that are not constant, we can use the get_support() method of the filter that we created. Execute the following script to see the number of non-constant features.

# len(train_features.columns[constant_filter.get_support()])
# # %%

# constant_filter.get_support()
# constant_columns = [column for column in train_features.columns
#                     if column not in train_features.columns[constant_filter.get_support()]]

# test_features.drop(labels=constant_columns, axis=1, inplace=True)
# # %%
# train_features = constant_filter.transform(train_features)
# test_features = constant_filter.transform(test_features)

# train_features.shape, test_features.shape

# #%%
# #  Constant filter

# constant_filter = VarianceThreshold(threshold=0)
# constant_filter.fit(train_features)

# len(train_features.columns[constant_filter.get_support()])

# constant_columns = [column for column in train_features.columns
#                     if column not in train_features.columns[constant_filter.get_support()]]

# train_features.drop(labels=constant_columns, axis=1, inplace=True)
# test_features.drop(labels=constant_columns, axis=1, inplace=True)

# #%%
# #  Quasi Constant filter

# qconstant_filter = VarianceThreshold(threshold=0.01)

# qconstant_filter.fit(train_features)

# qconstant_columns = [column for column in train_features.columns
#                     if column not in train_features.columns[qconstant_filter.get_support()]]

# print(len(qconstant_columns))
# # %%
# train_features = qconstant_filter.transform(train_features)
# test_features = qconstant_filter.transform(test_features)

# train_features.shape, test_features.shape
# # %%
# # Check duplicated features


# # %%
# train_features_T.duplicated()
# # %%

#     return train_features, test_features, train_labels, test_labels, correlated_features

# train_features, test_features, train_labels, test_labels, correlated_features = check_correlation(santandar_data,label_col="TARGET")
# # %%
# len(correlated_features)
# # %%
# train_features_post_correl = train_features.drop(labels=correlated_features,axis=1)
# # %%
# # #
# def create_stratified_subset(data,train_labels):

#     return(train_test_split(
#             data,
#             train_labels,
#             test_size=0.8,stratify=train_labels,
#             random_state=41))

# # sampledata = train_features[filtered_features]
# train_features_sub, _, train_labels_sub,_ =create_stratified_subset(train_features_post_correl,train_labels)
# #%%

# # %%

# # %%
# # %%


# #%%

# clf = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=3)
# clf.fit(train_features[filtered_features].fillna(0), train_labels)
# # %%
# clf = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=3)
# clf.fit(train_features[filtered_features].fillna(0), train_labels)

# train_pred = clf.predict_proba(train_features[filtered_features].fillna(0))
# print('Accuracy on training set: {}'.format(roc_auc_score(train_labels, train_pred[:,1])))

# test_pred = clf.predict_proba(test_features[filtered_features].fillna(0))
# print('Accuracy on test set: {}'.format(roc_auc_score(test_labels, test_pred [:,1])))
# # %%
# from mlxtend.feature_selection import ExhaustiveFeatureSelector
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.metrics import roc_auc_score

# feature_selector = ExhaustiveFeatureSelector(RandomForestClassifier(n_jobs=-1),
#            min_features=2,
#            max_features=4,
#            scoring='roc_auc',
#            print_progress=True,
#            cv=2)
# features = feature_selector.fit(np.array(train_features_sub[filtered_features].fillna(0)), train_labels_sub)
# # %%

# # %%
