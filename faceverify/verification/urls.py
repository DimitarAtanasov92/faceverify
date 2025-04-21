from django.urls import path
from .views import register, verify_user, upload_image, verification_success

urlpatterns = [
    path('', register, name='register'),
    path('verify/', verify_user, name='verify_user'),
    path('upload/', upload_image, name='upload_image'),
    path('success/', verification_success, name='verification_success'),
]