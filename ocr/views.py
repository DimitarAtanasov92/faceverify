from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import OCRImage
from .forms import OCRImageForm
from .ocr_utils import perform_ocr, ocr_with_confidence, extract_id_document_fields
import os

def ocr_home(request):
    """Display the home page with the OCR upload form"""
    if request.method == 'POST':
        form = OCRImageForm(request.POST, request.FILES)
        if form.is_valid():
            ocr_image = form.save(commit=False)
            ocr_image.save()
            
            # Get the image path
            image_path = ocr_image.image.path
            
            # Perform OCR with field extraction
            extracted_data = extract_id_document_fields(image_path)
            
            # Save all extracted data
            ocr_image.processed_text = extracted_data['full_text']
            ocr_image.surname = extracted_data['surname']
            ocr_image.name = extracted_data['name']
            ocr_image.fathers_name = extracted_data['fathers_name']
            ocr_image.date_of_birth = extracted_data['date_of_birth']
            ocr_image.date_of_expiry = extracted_data['date_of_expiry']
            ocr_image.document_number = extracted_data['document_number']
            ocr_image.date_of_issue = extracted_data['date_of_issue']
            ocr_image.save()
            
            return redirect('ocr:ocr_result', ocr_image.id)
    else:
        form = OCRImageForm()
    
    return render(request, 'ocr/ocr_home.html', {'form': form})

def ocr_result(request, image_id):
    """Display the OCR results for a processed image"""
    ocr_image = OCRImage.objects.get(id=image_id)
    return render(request, 'ocr/ocr_result.html', {'ocr_image': ocr_image})

@csrf_exempt
def api_ocr(request):
    """API endpoint for OCR processing"""
    if request.method == 'POST':
        form = OCRImageForm(request.POST, request.FILES)
        if form.is_valid():
            ocr_image = form.save()
            
            # Get the image path
            image_path = ocr_image.image.path
            
            # Perform OCR with field extraction
            extracted_data = extract_id_document_fields(image_path)
            
            # Save all extracted data
            ocr_image.processed_text = extracted_data['full_text']
            ocr_image.surname = extracted_data['surname']
            ocr_image.name = extracted_data['name']
            ocr_image.fathers_name = extracted_data['fathers_name']
            ocr_image.date_of_birth = extracted_data['date_of_birth']
            ocr_image.date_of_expiry = extracted_data['date_of_expiry']
            ocr_image.document_number = extracted_data['document_number']
            ocr_image.date_of_issue = extracted_data['date_of_issue']
            ocr_image.save()
            
            # Create a structured response
            response_data = {
                'success': True,
                'image_id': ocr_image.id,
                'full_text': extracted_data['full_text'],
                'structured_data': {
                    'surname': ocr_image.surname,
                    'name': ocr_image.name,
                    'fathers_name': ocr_image.fathers_name,
                    'date_of_birth': ocr_image.date_of_birth,
                    'date_of_expiry': ocr_image.date_of_expiry,
                    'document_number': ocr_image.document_number,
                    'date_of_issue': ocr_image.date_of_issue
                }
            }
            
            return JsonResponse(response_data)
        else:
            return JsonResponse({'success': False, 'errors': form.errors})
    
    return JsonResponse({'success': False, 'message': 'Only POST requests are supported'})

def recent_ocr_images(request):
    """Display recently processed OCR images"""
    images = OCRImage.objects.all().order_by('-uploaded_at')[:10]
    return render(request, 'ocr/recent_images.html', {'images': images})
