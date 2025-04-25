from django.urls import path
from .views import (
    start_verification_view,
    upload_selfie_view,
    verification_status_view,
    verification_failed_view,
    update_verification_data,
)

urlpatterns = [
    path('start/', start_verification_view, name='start_verification'),
    path('upload-selfie/', upload_selfie_view, name='upload_selfie'),
    path('status/', verification_status_view, name='verification_status'),
    path('failed/', verification_failed_view, name='verification_failed'),
    path('update-data/', update_verification_data, name='update_verification_data'),
]