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
    
    path('experiment', views.ExperimentListView.as_view(), name='all'),
    path('experiment/<int:pk>/detail', views.ExperimentDetailView.as_view(), name='experiment_detail'),
    path('experiment/create/', views.ExperimentCreateView.as_view(), name='experiment_create'),
    path('experiment/<int:pk>/update/', views.ExperimentUpdateView.as_view(), name='experiment_update'),
    path('experiment/<int:pk>/delete/', views.ExperimentDeleteView.as_view(), name='experiment_delete'),
]