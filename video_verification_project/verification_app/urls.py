from django.urls import path
from . import views

urlpatterns = [
    path('start_verification/', views.start_verification, name='start_verification'),
    path('verification_process/', views.verification_process, name='verification_process'),
    path('verification_status/', views.verification_status, name='verification_status'),
]