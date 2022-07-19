from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),  #the path for the index view
]