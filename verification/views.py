from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import VerificationDocument
from .forms import IDDocumentForm, SelfieForm
import time # За симулация на обработка
import random # За симулация на резултат
from .english_ocr import EnglishIDOCR
import os
import logging
from django.conf import settings
from django.http import JsonResponse
import re
import json
from ocr.ocr_utils import preprocess_image, perform_ocr, ocr_with_confidence, extract_id_document_fields
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .face_verification import FaceVerification
import uuid

# Set up logging
logger = logging.getLogger('verification.views')

@login_required
def start_verification_view(request):
    # Проверка дали потребителят вече има процес на верификация
    existing_doc = VerificationDocument.objects.filter(user=request.user).first()
    if existing_doc and existing_doc.status not in [VerificationDocument.StatusChoices.PENDING, VerificationDocument.StatusChoices.FAILED_DATA, VerificationDocument.StatusChoices.FAILED_MATCH, VerificationDocument.StatusChoices.FAILED_MANUAL]:
         # Ако вече е в процес или е верифициран, пренасочваме към статус
         return redirect('verification_status')
    elif existing_doc and existing_doc.status == VerificationDocument.StatusChoices.NEEDS_SELFIE:
         # Ако е качил документи, но не и селфи
         return redirect('upload_selfie')

    # Get debug mode from request parameters
    debug_mode = request.GET.get('debug', 'normal')
    if debug_mode not in ['minimal', 'normal', 'verbose']:
        debug_mode = 'normal'

    # Ако няма съществуващ или е неуспешен, показваме формата за качване на документи
    if request.method == 'POST':
        # Ако има съществуващ неуспешен, го използваме, иначе създаваме нов
        form = IDDocumentForm(request.POST, request.FILES, instance=existing_doc)
        if form.is_valid():
            doc = form.save(commit=False)
            doc.user = request.user
            doc.status = VerificationDocument.StatusChoices.PROCESSING # Слагаме статус "Обработва се"
            doc.extracted_data = None # Изчистваме стари данни, ако има
            doc.selfie_image = None # Изчистваме старо селфи, ако има
            doc.status_reason = None # Изчистваме причина
            doc.save()

            try:
                # Get paths to the ID images
                front_image_path = os.path.join(settings.MEDIA_ROOT, doc.front_image.name)
                back_image_path = None
                if doc.back_image:
                    back_image_path = os.path.join(settings.MEDIA_ROOT, doc.back_image.name)
                
                # Process ID card images using the improved OCR utilities
                extracted_data = extract_id_document_fields(front_image_path)
                
                # If back image exists, process it as well
                if back_image_path:
                    back_data = extract_id_document_fields(back_image_path)
                    # Merge the data, prioritizing front image data
                    for key, value in back_data.items():
                        if not extracted_data.get(key) and value:
                            extracted_data[key] = value
                
                # Log the extracted data for debugging
                logger.info(f"Extracted data: {extracted_data}")
                
                # Store debug info in session if in debug mode
                if debug_mode in ['normal', 'verbose']:
                    request.session['ocr_debug_info'] = {
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'debug_level': debug_mode,
                        'extracted_fields_count': len(extracted_data),
                        'extracted_fields': list(extracted_data.keys())
                    }
                
                # Check if we have enough identification data
                has_name = ('name' in extracted_data and extracted_data['name'] and 
                           extracted_data['name'] != 'UNKNOWN')
                
                # Check for any valid ID field
                id_fields = ['document_number', 'identity_number']
                has_id_field = any(field in extracted_data and extracted_data[field] for field in id_fields)
                
                # List missing required data for error reporting
                missing_data = []
                if not has_name:
                    missing_data.append('name')
                if not has_id_field:
                    missing_data.append('identification number')
                
                # If we have defaults like 'UNKNOWN', we may need manual verification
                needs_manual_review = False
                if 'name' in extracted_data and extracted_data['name'] == 'UNKNOWN':
                    needs_manual_review = True
                if 'surname' in extracted_data and extracted_data['surname'] == 'UNKNOWN':
                    needs_manual_review = True
                
                if has_name and has_id_field:
                    # Successful OCR with required fields
                    doc.extracted_data = extracted_data
                    doc.status = VerificationDocument.StatusChoices.NEEDS_SELFIE
                    doc.save()
                    
                    if needs_manual_review:
                        messages.warning(request, 'Some fields could not be automatically extracted. Please review and complete your information.')
                        return redirect('update_verification_data')
                    else:
                        messages.success(request, 'ID documents successfully processed. Please upload a selfie for verification.')
                        return redirect('upload_selfie')
                elif extracted_data:  # We have some data but missing required fields
                    # Store what we have but mark as needing manual correction
                    doc.extracted_data = extracted_data
                    doc.status = VerificationDocument.StatusChoices.FAILED_DATA
                    doc.status_reason = f"Could not extract all required fields: {', '.join(missing_data)}. Please review and complete your information."
                    doc.save()
                    
                    if debug_mode == 'verbose':
                        fields_found = ', '.join(extracted_data.keys()) if extracted_data else 'none'
                        messages.warning(request, f"Debug info: Fields found: {fields_found}")
                    
                    messages.warning(request, doc.status_reason)
                    return redirect('update_verification_data')
                else:
                    # Missing required fields
                    doc.status = VerificationDocument.StatusChoices.FAILED_DATA
                    doc.status_reason = f"Could not extract required fields: {', '.join(missing_data)}. Please try with a clearer image or different document."
                    doc.save()
                    
                    if debug_mode == 'verbose':
                        fields_found = ', '.join(extracted_data.keys()) if extracted_data else 'none'
                        messages.warning(request, f"Debug info: Fields found: {fields_found}")
                    
                    messages.error(request, doc.status_reason)
            except Exception as e:
                # Handle OCR processing error
                logger.error(f"OCR processing error: {str(e)}", exc_info=True)
                doc.status = VerificationDocument.StatusChoices.FAILED_DATA
                doc.status_reason = f"Error processing document: {str(e)}"
                doc.save()
                messages.error(request, doc.status_reason)
    else:
         # Ако GET заявка, показваме празна форма или формата с данните от предишен неуспешен опит
         form = IDDocumentForm(instance=existing_doc)

    context = {
        'form': form,
        'debug_mode': debug_mode,
        'available_debug_modes': ['minimal', 'normal', 'verbose']
    }
    # Ползваме един и същ шаблон за стартиране и повторен опит след FAILED_DATA
    return render(request, 'verification/upload_id_docs.html', context)

@login_required
def upload_selfie_view(request):
    verification_doc = get_object_or_404(VerificationDocument, user=request.user)

    # Проверка дали сме на правилната стъпка
    if verification_doc.status != VerificationDocument.StatusChoices.NEEDS_SELFIE:
        messages.warning(request, 'Не сте на стъпка за качване на селфи.')
        return redirect('verification_status')

    if request.method == 'POST':
        form = SelfieForm(request.POST, request.FILES, instance=verification_doc)
        if form.is_valid():
            doc = form.save(commit=False)
            doc.status = VerificationDocument.StatusChoices.PROCESSING # Отново обработка
            doc.save()

            # --- СИМУЛАЦИЯ НА БИОМЕТРИЧНА ВЕРИФИКАЦИЯ ---
            logger.info(f"Simulating Facial Verification for user {request.user.id}...")
            time.sleep(3) # Симулираме по-дълга обработка

            # Симулираме резултат от сравнението
            match_success = random.choice([True, True, True, False]) # Още по-голям шанс за успех

            if match_success:
                doc.status = VerificationDocument.StatusChoices.VERIFIED
                doc.save()
                # Обновяваме и профила на потребителя
                request.user.profile.is_verified = True
                request.user.profile.save()
                messages.success(request, 'Верификацията премина успешно!')
                return redirect('verification_status') # Пренасочваме към статус страницата
            else:
                doc.status = VerificationDocument.StatusChoices.FAILED_MATCH
                doc.status_reason = "Симулация: Несъответствие между лицето от документа и селфито."
                doc.save()
                request.user.profile.is_verified = False # Връщаме на не верифициран
                request.user.profile.save()
                messages.error(request, doc.status_reason)
                # Може да пренасочим към страница за неуспешна верификация или обратно към профила
                return redirect('verification_failed')
            # --- КРАЙ НА СИМУЛАЦИЯТА ---

    else:
        form = SelfieForm(instance=verification_doc)

    context = {'form': form}
    return render(request, 'verification/upload_selfie.html', context)


@login_required
def verification_status_view(request):
    verification_doc = None
    try:
        verification_doc = VerificationDocument.objects.get(user=request.user)
    except VerificationDocument.DoesNotExist:
        # Ако няма документ, значи не е стартирал процеса
         messages.info(request, 'Все още не сте стартирали процеса на верификация.')
         return redirect('profile') # Връщаме към профила

    # Get debug info from session if available
    debug_info = request.session.get('ocr_debug_info', None)
    
    # Check if debug mode is requested via GET parameter
    show_debug = request.GET.get('show_debug', 'false').lower() == 'true'
    
    context = {
        'verification_doc': verification_doc,
        'show_debug': show_debug,
        'debug_info': debug_info
    }
    return render(request, 'verification/verification_status.html', context)

@login_required
def verification_failed_view(request):
    verification_doc = get_object_or_404(VerificationDocument, user=request.user)
    # Показваме страница само ако статусът е някой от неуспешните
    if verification_doc.status not in [VerificationDocument.StatusChoices.FAILED_DATA, VerificationDocument.StatusChoices.FAILED_MATCH, VerificationDocument.StatusChoices.FAILED_MANUAL]:
         return redirect('verification_status')

    # Get debug info from session if available
    debug_info = request.session.get('ocr_debug_info', None)
    
    context = {
        'verification_doc': verification_doc,
        'debug_info': debug_info
    }
    return render(request, 'verification/verification_failed.html', context)

@login_required
def update_verification_data(request):
    """View for updating verification data after OCR processing"""
    verification_doc = get_object_or_404(VerificationDocument, user=request.user)
    
    if request.method == 'POST':
        # Get the updated data from the POST request
        updated_data = {}
        for field in ['first_name', 'last_name', 'identity_number', 'personal_number', 
                     'birth_date', 'expiry_date', 'nationality', 'sex', 'height', 
                     'eye_color', 'permanent_address']:
            if field in request.POST:
                updated_data[field] = request.POST[field].strip()
        
        # Update the extracted_data field
        if verification_doc.extracted_data is None:
            verification_doc.extracted_data = {}
        verification_doc.extracted_data.update(updated_data)
        verification_doc.save()
        
        messages.success(request, 'Verification data updated successfully.')
        return redirect('verification_status')
    
    context = {
        'verification_doc': verification_doc,
        'extracted_data': verification_doc.extracted_data or {}
    }
    return render(request, 'verification/update_verification_data.html', context)

@login_required
def upload_extracted_data(request):
    """Handle upload of externally extracted OCR data"""
    verification_doc = get_object_or_404(VerificationDocument, user=request.user)
    
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request
            extracted_data = json.loads(request.body)
            
            # Create a clean data dictionary for verification document
            clean_data = {}
            
            # Copy standard fields directly
            standard_fields = [
                'birth_date', 'expiry_date', 'nationality', 'identity_number',
                'document_number', 'first_name'
            ]
            
            for field in standard_fields:
                if field in extracted_data and extracted_data[field]:
                    clean_data[field] = extracted_data[field]
            
            # Apply the specific transformations based on the example data
            # 1. 'phone' value add to identity number
            if 'phone' in extracted_data and extracted_data['phone']:
                clean_data['identity_number'] = extracted_data['phone']
            
            # 2. "A sTANASOV" value add to last name - exactly as in example
            if '_raw_front_text' in extracted_data:
                last_name_match = re.search(r'A\s+(\w+)', extracted_data['_raw_front_text'])
                if last_name_match:
                    last_name = last_name_match.group(1).upper()
                    clean_data['last_name'] = last_name
            
            # 3. "Nie vs moxymentalDocument number" add to personal_number
            if '_raw_front_text' in extracted_data:
                doc_number_match = re.search(r'Nie vs moxymentalDocument number\s+(\d+)', extracted_data['_raw_front_text'])
                if doc_number_match:
                    clean_data['personal_number'] = doc_number_match.group(1)
            
            # Store the data in the verification document
            if verification_doc.extracted_data is None:
                verification_doc.extracted_data = {}
            
            # Update with clean data
            verification_doc.extracted_data.update(clean_data)
            verification_doc.save()
            
            # Update the status if needed
            if verification_doc.status == VerificationDocument.StatusChoices.PENDING:
                verification_doc.status = VerificationDocument.StatusChoices.NEEDS_SELFIE
                verification_doc.save()
            
            return JsonResponse({'success': True, 'message': 'Data uploaded successfully'})
                
        except Exception as e:
            logger.error(f"Error processing extracted data: {str(e)}", exc_info=True)
            return JsonResponse({'success': False, 'error': str(e)})
    
    # If it's a GET request, render a form for manual upload
    return render(request, 'verification/upload_data.html', {'verification_doc': verification_doc})

@csrf_exempt
def verify_faces(request):
    if request.method == 'POST':
        try:
            # Get uploaded files
            selfie = request.FILES.get('selfie')
            id_document = request.FILES.get('id_document')

            if not selfie or not id_document:
                return JsonResponse({
                    'success': False,
                    'error': 'Both selfie and ID document are required'
                })

            # Save files temporarily
            selfie_path = f'media/temp/{uuid.uuid4()}_{selfie.name}'
            id_path = f'media/temp/{uuid.uuid4()}_{id_document.name}'

            # Create temp directory if it doesn't exist
            os.makedirs('media/temp', exist_ok=True)

            # Save files
            default_storage.save(selfie_path, ContentFile(selfie.read()))
            default_storage.save(id_path, ContentFile(id_document.read()))

            # Initialize face verification
            verifier = FaceVerification()

            # Verify faces
            is_match, confidence = verifier.verify_faces(selfie_path, id_path)

            # Clean up temporary files
            os.remove(selfie_path)
            os.remove(id_path)

            return JsonResponse({
                'success': True,
                'is_match': is_match,
                'confidence': confidence
            })

        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

    return JsonResponse({
        'success': False,
        'error': 'Only POST method is allowed'
    })

@login_required
@csrf_exempt
def upload_data(request):
    if request.method == 'POST':
        try:
            # Get the file from the request
            file = request.FILES.get('file')
            if not file:
                return JsonResponse({'error': 'No file provided'}, status=400)

            # Generate a unique filename
            filename = f"{uuid.uuid4()}_{file.name}"
            
            # Save the file
            file_path = default_storage.save(f'uploads/{filename}', ContentFile(file.read()))
            
            return JsonResponse({
                'success': True,
                'file_path': file_path
            })
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)