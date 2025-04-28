from django.urls import path
from . import views

app_name = 'ocr'

urlpatterns = [
    path('', views.ocr_home, name='ocr_home'),
    path('result/<int:image_id>/', views.ocr_result, name='ocr_result'),
    path('api/ocr/', views.api_ocr, name='api_ocr'),
    path('recent/', views.recent_ocr_images, name='recent_images'),
] 