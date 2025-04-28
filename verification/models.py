from django.db import models
from django.contrib.auth.models import User
import os
import uuid

# Функция за генериране на уникално име на файл
def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/verification_docs/user_<id>/<uuid>.<ext>
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('verification_docs', f'user_{instance.user.id}', filename)

class VerificationDocument(models.Model):
    class StatusChoices(models.TextChoices):
        PENDING = 'PENDING', 'Изчаква качване'
        PROCESSING = 'PROCESSING', 'Обработва се'
        NEEDS_SELFIE = 'NEEDS_SELFIE', 'Нужно е селфи/видео'
        VERIFIED = 'VERIFIED', 'Верифициран'
        FAILED_DATA = 'FAILED_DATA', 'Неуспешно извличане на данни'
        FAILED_MATCH = 'FAILED_MATCH', 'Несъответствие на лице'
        FAILED_MANUAL = 'FAILED_MANUAL', 'Отхвърлен ръчно'

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='verification_document')
    document_type = models.CharField(max_length=50, blank=True, null=True, choices=[('ID_CARD', 'Лична карта'), ('PASSPORT', 'Паспорт')]) # Примерни типове
    front_image = models.ImageField(upload_to=user_directory_path)
    back_image = models.ImageField(upload_to=user_directory_path, blank=True, null=True) # Не всички документи имат гръб
    selfie_image = models.ImageField(upload_to=user_directory_path, blank=True, null=True) # Или FileField за видео
    # verification_video = models.FileField(upload_to=user_directory_path, blank=True, null=True)

    extracted_data = models.JSONField(blank=True, null=True) # Тук ще симулираме извлечените данни
    status = models.CharField(max_length=20, choices=StatusChoices.choices, default=StatusChoices.PENDING)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    status_reason = models.TextField(blank=True, null=True) # Причина при неуспех

    def __str__(self):
        return f"Verification for {self.user.username} - Status: {self.get_status_display()}"
