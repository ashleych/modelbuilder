from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('', views.TraindataListView.as_view(), name='all'),
    path('traindata/<int:pk>/detail', views.TraindataDetailView.as_view(), name='traindata_detail'),
    path('traindata/create/', views.TraindataCreateView.as_view(), name='traindata_create'),
    path('traindata/<int:pk>/update/', views.TraindataUpdateView.as_view(), name='traindata_update'),
    path('traindata/<int:pk>/delete/', views.TraindataDeleteView.as_view(), name='traindata_delete'),
    path('upload/csv', views.upload_csv, name='upload_csv'),
]