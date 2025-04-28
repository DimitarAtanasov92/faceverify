from django import forms
from .models import OCRImage

class OCRImageForm(forms.ModelForm):
    class Meta:
        model = OCRImage
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'})
        } 