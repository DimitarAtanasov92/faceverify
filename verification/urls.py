from django.urls import path
from .views import (
    start_verification_view,
    upload_selfie_view,
    verification_status_view,
    verification_failed_view,
    update_verification_data,
    upload_extracted_data,
    upload_data,
    verify_faces,
)

urlpatterns = [
    path('start/', start_verification_view, name='start_verification'),
    path('upload-selfie/', upload_selfie_view, name='upload_selfie'),
    path('status/', verification_status_view, name='verification_status'),
    path('failed/', verification_failed_view, name='verification_failed'),
    path('update-data/', update_verification_data, name='update_verification_data'),
    path('upload-extracted-data/', upload_extracted_data, name='upload_extracted_data'),
    path('upload/', upload_data, name='upload_data'),
    path('verify-faces/', verify_faces, name='verify_faces'),
]