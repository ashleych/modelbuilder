from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/csv', views.upload_csv, name='upload_csv'),
    path('traindata', views.TraindataListView.as_view(), name='all'),
    path('traindata/<int:pk>/detail', views.TraindataDetailView.as_view(), name='traindata_detail'),
    path('traindata/create/', views.TraindataCreateView.as_view(), name='traindata_create'),
    path('traindata/<int:pk>/update/', views.TraindataUpdateView.as_view(), name='traindata_update'),
    path('traindata/<int:pk>/delete/', views.TraindataDeleteView.as_view(), name='traindata_delete'),
    
    path('experiment', views.ExperimentListView.as_view(), name='experiment_all'),
    path('experiment/<int:pk>/detail', views.ExperimentDetailView.as_view(), name='experiment_detail'),
    path('experiment/create/', views.ExperimentCreateView.as_view(), name='experiment_create'),
    # path('experiment/start/', views.experiment_start, name='experiment_start'),
    path('experiment/start/', views.ExperimentFormView.as_view(), name='experiment_form'),
    path('experiment/<int:pk>/update/', views.ExperimentUpdateView.as_view(), name='experiment_update'),
    path('experiment/<int:pk>/delete/', views.ExperimentDeleteView.as_view(), name='experiment_delete'),

    path('stationarity', views.StationarityListView.as_view(), name='stationarity_all'),
    path('stationarity/<int:pk>/detail', views.StationarityDetailView.as_view(), name='stationarity_detail'),
    path('stationarity/create/', views.StationarityCreateView.as_view(), name='stationarity_create'),
    path('stationarity/<int:pk>/update/', views.StationarityUpdateView.as_view(), name='stationarity_update'),
    path('stationarity/<int:pk>/delete/', views.StationarityDeleteView.as_view(), name='stationarity_delete'),

    path('variables/<int:experiment_id>', views.VariablesListView.as_view(), name='variables'),
    path('variables/<int:pk>/detail', views.VariablesDetailView.as_view(), name='variables_detail'),
    path('variables/create/', views.VariablesCreateView.as_view(), name='variables_create'),
    path('variables/<int:pk>/update/', views.VariablesUpdateView.as_view(), name='variables_update'),
    path('variables/<int:pk>/delete/', views.VariablesDeleteView.as_view(), name='variables_delete'),
    
    path('manualvariableselection', views.ManualvariableselectionListView.as_view(), name='all'),
    path('manualvariableselection/<int:pk>/detail', views.ManualvariableselectionDetailView.as_view(), name='manualvariableselection_detail'),
    path('manualvariableselection/create/', views.ManualvariableselectionCreateView.as_view(), name='manualvariableselection_create'),
    path('manualvariableselection/<int:pk>/update/', views.ManualvariableselectionUpdateView.as_view(), name='manualvariableselection_update'),
    path('manualvariableselection/<int:pk>/delete/', views.ManualvariableselectionDeleteView.as_view(), name='manualvariableselection_delete'),
    
    path('classificationmodel', views.ClassificationmodelListView.as_view(), name='all'),
    path('classificationmodel/<int:pk>/detail', views.ClassificationmodelDetailView.as_view(), name='classificationmodel_detail'),
    path('classificationmodel/create/', views.ClassificationmodelCreateView.as_view(), name='classificationmodel_create'),
    path('classificationmodel/<int:pk>/update/', views.ClassificationmodelUpdateView.as_view(), name='classificationmodel_update'),
    path('classificationmodel/<int:pk>/delete/', views.ClassificationmodelDeleteView.as_view(), name='classificationmodel_delete'),
    path('index_j/', views.index_j, name='index_j')
]