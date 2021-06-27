from django.urls import path

from . import views

app_name = 'trash'

urlpatterns = [
    path('', views.main, name='trash main'),
    path('trashclassify', views.trashClassify, name='trash classify')
]