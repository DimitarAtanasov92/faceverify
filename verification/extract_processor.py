import re
import logging

logger = logging.getLogger(__name__)

def process_extracted_data(data):
    """
    Process and clean extracted data from OCR
    
    Args:
        data: Dictionary with raw extracted data
        
    Returns:
        Dictionary with processed data
    """
    if not data:
        return data
    
    # Make a copy to avoid modifying the original
    processed = data.copy()
    
    # Process last_name from "A sTANASOV" to "ATANASOV"
    if 'last_name' in processed and processed['last_name']:
        last_name = processed['last_name']
        # Remove lowercase 's', common OCR error in Cyrillic names
        if re.search(r'^\s*[A-Z]\s+s[A-Z]+', last_name):
            last_name = last_name.replace('s', '')
        
        # Remove spaces and ensure uppercase
        last_name = re.sub(r'\s+', '', last_name).upper()
        processed['last_name'] = last_name
    
    # Map phone to identity_number if phone exists and identity_number doesn't
    if 'phone' in processed and processed['phone'] and 'identity_number' not in processed:
        processed['identity_number'] = processed['phone']
        logger.info(f"Mapped phone to identity_number: {processed['phone']}")
    
    # Map document_number from "Nie vs moxymentalDocument number" to personal_number
    if 'document_number' in processed and processed['document_number'] and 'personal_number' not in processed:
        processed['personal_number'] = processed['document_number']
        logger.info(f"Mapped document_number to personal_number: {processed['document_number']}")
    
    return processed 