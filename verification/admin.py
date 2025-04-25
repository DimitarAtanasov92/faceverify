from django.contrib import admin
from .models import VerificationDocument

@admin.register(VerificationDocument)
class VerificationDocumentAdmin(admin.ModelAdmin):
    list_display = ('user', 'document_type', 'status', 'created_at', 'updated_at')
    list_filter = ('status', 'document_type', 'created_at')
    search_fields = ('user__username', 'user__email')
    readonly_fields = ('created_at', 'updated_at', 'extracted_data')
