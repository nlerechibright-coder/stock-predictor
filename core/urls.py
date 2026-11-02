from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('predict/', views.generate_prediction, name='generate_prediction'),
    path('history/', views.prediction_history, name='prediction_history'),
    path('methodology/', views.methodology, name='methodology'),
]
