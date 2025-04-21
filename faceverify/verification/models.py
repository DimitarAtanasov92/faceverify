from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    is_verified = models.BooleanField(default=False)
    id_document_front = models.ImageField(upload_to='id_documents/')
    id_document_back = models.ImageField(upload_to='id_documents/')