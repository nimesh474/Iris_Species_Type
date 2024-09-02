from django.urls import path
from iris_type_prediction import views

urlpatterns = [
    path('', views.predict_type, name='predict_type')
]