from django.contrib import admin
from .models import OCRImage

@admin.register(OCRImage)
class OCRImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'uploaded_at', 'processed_text_preview')
    list_filter = ('uploaded_at',)
    search_fields = ('processed_text',)
    readonly_fields = ('uploaded_at',)
    
    def processed_text_preview(self, obj):
        if obj.processed_text:
            return obj.processed_text[:100] + '...' if len(obj.processed_text) > 100 else obj.processed_text
        return 'No text extracted'
    processed_text_preview.short_description = 'Extracted Text'
