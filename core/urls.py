from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('generate/', views.generate_prediction, name='generate_prediction'),
    path('history/', views.prediction_history, name='prediction_history'),
    path('methodology/', views.methodology, name='methodology'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('export/pdf/<int:record_id>/', views.export_prediction_pdf, name='export_prediction_pdf'),
    path('models/', views.model_management, name='model_management'),
    path('models/activate/<int:model_id>/', views.activate_model, name='activate_model'),
    path('performance/', views.performance_tracking, name='performance_tracking'),
]
