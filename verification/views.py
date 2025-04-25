from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import VerificationDocument
from .forms import IDDocumentForm, SelfieForm
import time # За симулация на обработка
import random # За симулация на резултат
from .english_ocr import EnglishIDOCR
from .simple_ocr import SimpleOCR  # Import the SimpleOCR class instead of AI OCR
from .extract_processor import process_extracted_data  # Import our new processing function
import os
import logging
from django.conf import settings

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
        
    # Get OCR engine parameter
    ocr_engine = request.GET.get('ocr_engine', 'advanced')  # Default to advanced OCR
    if ocr_engine not in ['advanced', 'standard']:
        ocr_engine = 'advanced'

    # Обработваме формата и файла, ако е POST заявка
    if request.method == 'POST':
        if existing_doc:
            form = IDDocumentForm(request.POST, request.FILES, instance=existing_doc)
        else:
            form = IDDocumentForm(request.POST, request.FILES)

        if form.is_valid():
            # Запазваме документа с начален статус
            doc = form.save(commit=False)
            doc.user = request.user
            doc.status = VerificationDocument.StatusChoices.PROCESSING
            doc.save()
            
            messages.info(request, 'Processing your ID documents...')

            try:
                # Get paths to the ID images
                front_image_path = os.path.join(settings.MEDIA_ROOT, doc.front_image.name)
                back_image_path = None
                if doc.back_image:
                    back_image_path = os.path.join(settings.MEDIA_ROOT, doc.back_image.name)
                
                extracted_data = {}
                
                # Choose OCR engine based on the parameter
                if ocr_engine == 'advanced':
                    # Use the advanced OCR engine (SimpleOCR)
                    logger.info(f"Using advanced OCR engine for user {request.user.id}")
                    
                    # Initialize SimpleOCR with debug mode
                    simple_ocr = SimpleOCR(languages=['eng'], debug_mode=(debug_mode != 'minimal'))
                    
                    # Process ID card images
                    extracted_data = simple_ocr.process_id_card(
                        front_image_path, 
                        back_image_path, 
                        debug_level=debug_mode
                    )
                else:
                    # Use the standard English OCR engine
                    logger.info(f"Using standard OCR engine for user {request.user.id}")
                    
                    # Initialize English ID OCR with debug mode
                    ocr = EnglishIDOCR(debug_mode=(debug_mode != 'minimal'))
                    
                    # Process ID card images with debug level
                    extracted_data = ocr.process_id_card(front_image_path, back_image_path, debug_level=debug_mode)
                
                # Process the extracted data to apply our custom mappings
                extracted_data = process_extracted_data(extracted_data)
                
                # Log the extracted data for debugging
                logger.info(f"Extracted data after processing: {extracted_data}")
                
                # Store debug info in session if in debug mode
                if debug_mode in ['normal', 'verbose']:
                    request.session['ocr_debug_info'] = {
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'debug_level': debug_mode,
                        'ocr_engine': ocr_engine,
                        'extracted_fields_count': len(extracted_data),
                        'extracted_fields': list(extracted_data.keys())
                    }
                
                # Check if we have enough identification data - more flexible validation
                # We need at least a name and ONE of several possible ID fields
                has_name = ('first_name' in extracted_data and extracted_data['first_name'] and 
                           extracted_data['first_name'] != 'UNKNOWN')
                
                # Check for any valid ID field (we accept multiple possibilities)
                id_fields = ['identity_number', 'document_number', 'personal_number', 'id_number']
                has_id_field = any(field in extracted_data and extracted_data[field] for field in id_fields)
                
                # List missing required data for error reporting
                missing_data = []
                if not has_name:
                    missing_data.append('name')
                if not has_id_field:
                    missing_data.append('identification number')
                
                # If we have defaults like 'UNKNOWN', we may need manual verification
                needs_manual_review = False
                if 'first_name' in extracted_data and extracted_data['first_name'] == 'UNKNOWN':
                    needs_manual_review = True
                if 'last_name' in extracted_data and extracted_data['last_name'] == 'UNKNOWN':
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
        'ocr_engine': ocr_engine,
        'available_debug_modes': ['minimal', 'normal', 'verbose'],
        'available_ocr_engines': ['advanced', 'standard']
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
        
        # Process the updated data with our custom processor
        processed_data = process_extracted_data(updated_data)
        
        # Update the extracted_data field
        if verification_doc.extracted_data is None:
            verification_doc.extracted_data = {}
        verification_doc.extracted_data.update(processed_data)
        verification_doc.save()
        
        messages.success(request, 'Verification data updated successfully.')
        return redirect('verification_status')
    
    context = {
        'verification_doc': verification_doc,
        'extracted_data': verification_doc.extracted_data or {}
    }
    return render(request, 'verification/update_verification_data.html', context)