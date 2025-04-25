from django import forms
from .models import VerificationDocument

class IDDocumentForm(forms.ModelForm):
    class Meta:
        model = VerificationDocument
        fields = ['document_type', 'front_image', 'back_image']
        widgets = {
            'document_type': forms.Select(attrs={'class': 'form-control'}),
            'front_image': forms.ClearableFileInput(attrs={'class': 'form-control'}),
            'back_image': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }
        labels = {
            'document_type': 'Тип документ',
            'front_image': 'Лицева страна на документа',
            'back_image': 'Гръб на документа (ако е приложимо)',
        }

class SelfieForm(forms.ModelForm):
     class Meta:
        model = VerificationDocument
        fields = ['selfie_image'] # Или 'verification_video'
        widgets = {
            'selfie_image': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        }
        labels = {
            'selfie_image': 'Качете Ваше селфи (или кратко видео)',
        }