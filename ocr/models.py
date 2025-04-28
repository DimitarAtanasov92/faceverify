from django.db import models

# Create your models here.

class OCRImage(models.Model):
    image = models.ImageField(upload_to='ocr_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_text = models.TextField(blank=True)
    
    # Document fields
    surname = models.CharField(max_length=100, blank=True)
    name = models.CharField(max_length=100, blank=True)
    fathers_name = models.CharField(max_length=100, blank=True)
    date_of_birth = models.CharField(max_length=50, blank=True)
    date_of_expiry = models.CharField(max_length=50, blank=True)
    document_number = models.CharField(max_length=50, blank=True)
    date_of_issue = models.CharField(max_length=50, blank=True)
    
    def __str__(self):
        return f"OCR Image {self.id} - {self.uploaded_at.strftime('%Y-%m-%d %H:%M')}"
