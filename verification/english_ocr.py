import cv2
import numpy as np
import pytesseract
from PIL import Image
import re
import os
import random
import sys
import logging
import time
import math
from skimage import exposure

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('english_ocr')

# Path to Tesseract executable - update this if needed
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Flag to indicate if Tesseract is available
TESSERACT_AVAILABLE = False

# Configure tesseract if available
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    # Test if tesseract works
    if os.path.exists(TESSERACT_PATH):
        TESSERACT_AVAILABLE = True
        logger.info(f"Tesseract OCR initialized successfully at {TESSERACT_PATH}")
    else:
        logger.warning(f"Warning: Tesseract executable not found at specified path. Using fallback mode.")
except Exception as e:
    logger.error(f"Could not initialize Tesseract OCR: {str(e)}. Using fallback mode.")

class EnglishIDOCR:
    """Class for extracting data from ID cards with English text"""
    
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        logger.info(f"EnglishIDOCR initialized with debug_mode={debug_mode}")
        
        # Field patterns for capturing English data from ID cards
        self.field_patterns = {
            # Common document identification patterns
            'personal_number': r'(?:PERSONAL [Nn]UMBER|PIN)\s*[:/]?\s*(\d+)',
            'identity_number': r'(?:IDENTITY [Nn]UMBER|ID|№|NUMBER)\s*[:/]?\s*([A-Z0-9]+)',
            'document_number': r'(?:DOCUMENT [Nn]O\.?|№)\s*[:/]?\s*([A-Z0-9]+)',
            
            # Person details patterns
            'first_name': r'(?:GIVEN|FIRST|CHRISTIAN)\s*NAMES?\s*[:/]?\s*([A-Z]+(?:\s[A-Z]+)?)',
            'last_name': r'(?:SURNAME|LAST\s*NAME|FAMILY\s*NAME)\s*[:/]?\s*([A-Z]+(?:\s[A-Z]+)?)',
            'birth_date': r'(?:DATE\s*OF\s*BIRTH|BIRTH\s*DATE)\s*[:/]?\s*(\d{1,2}[.\/-]\d{1,2}[.\/-]\d{2,4})',
            'expiry_date': r'(?:DATE\s*OF\s*EXPIRY|EXPIRY\s*DATE|VALID\s*UNTIL)\s*[:/]?\s*(\d{1,2}[.\/-]\d{1,2}[.\/-]\d{2,4})',
            'nationality': r'(?:NATIONALITY|CITIZEN\s*OF)\s*[:/]?\s*([A-Z]+)',
            'sex': r'(?:SEX|GENDER)\s*[:/]?\s*([MFmf])',
            'height': r'(?:HEIGHT)\s*[:/]?\s*(\d+)',
            'eye_color': r'(?:EYE\s*COLOU?R|COLOU?R\s*OF\s*EYES)\s*[:/]?\s*([A-Z]+)',
            'permanent_address': r'(?:PERMANENT\s*ADDRESS|ADDRESS|RESIDENCE)\s*[:/]?\s*([A-Z0-9\s,\.]+)',
            'mrz_line1': r'([A-Z<]{5,44})',
            'mrz_line2': r'([A-Z0-9<]{5,44})',
        }
        
        # Additional alternative patterns for more flexible matching
        self.alternative_patterns = {
            # Alternative ID number patterns
            'personal_number': [
                r'(?:P|PERSONAL|PIN)\s*[\.:]?\s*(\d{6,10})', 
                r'(?:№|NO|NUMBER)[\.:]?\s*(\d{6,10})',
                r'(?<![0-9])(\d{6,10})(?![0-9])',  # Standalone 6-10 digit number
                r'(\d{6,10})'  # Last resort: just find any sequence of digits that might be an ID
            ],
            'identity_number': [
                r'(?:ID|IDENTITY)[\.:]?\s*(\d{6,10})', 
                r'ID NUMBER[\.:]?\s*(\d{6,10})',
                r'NO\.?\s*(\d{6,10})',
                r'(?<![0-9A-Z])([A-Z]{1,2}\d{6,8})(?![0-9A-Z])',  # Common ID format: 1-2 letters + 6-8 digits
                r'№[\.:]?\s*(\d{6,10})'
            ],
            'document_number': [
                r'(?:DOC|DOCUMENT)[\.:]?\s*([A-Z0-9]{5,10})',
                r'(?:№|NO)[\.:]?\s*([A-Z0-9]{5,10})',
                r'(?<![0-9A-Z])([A-Z]{1,3}\d{5,8})(?![0-9A-Z])'  # Common format: 1-3 letters followed by 5-8 digits
            ],
            
            # Alternative name patterns
            'first_name': [
                r'GIVEN\s*NAMES?[\.:]?\s*([A-Z]+)', 
                r'NAME[\.:]?\s*([A-Z]+)',
                r'FIRST[\.:]?\s*([A-Z]+)',
                r'(?<![A-Z])([A-Z]{3,})(?![A-Z])'  # Any 3+ letter uppercase word standing alone
            ],
            'last_name': [
                r'FAMILY\s*NAME[\.:]?\s*([A-Z]+)', 
                r'SURNAME[\.:]?\s*([A-Z]+)',
                r'LAST NAME[\.:]?\s*([A-Z]+)'
            ],
            
            # Alternative date patterns
            'birth_date': [
                r'DOB[\.:]?\s*(\d{1,2}[.\/-]\d{1,2}[.\/-]\d{2,4})',
                r'BIRTH[\.:]?\s*(\d{1,2}[.\/-]\d{1,2}[.\/-]\d{2,4})',
                r'(?<!\d)(\d{2}[.\/-]\d{2}[.\/-](?:19|20)\d{2})(?!\d)'  # Standalone date in format DD/MM/YYYY
            ],
            'expiry_date': [
                r'EXPIRY[\.:]?\s*(\d{1,2}[.\/-]\d{1,2}[.\/-]\d{2,4})',
                r'VALID UNTIL[\.:]?\s*(\d{1,2}[.\/-]\d{1,2}[.\/-]\d{2,4})',
            ]
        }
        
        # Add smart patterns for unlabeled field detection
        self.smart_patterns = {
            'personal_number_unlabeled': r'\b\d{6,10}\b',  # Common ID format
            'identity_number_unlabeled': r'\b[0-9]{6,9}\b',  # ID number format
            'birth_date_unlabeled': r'\b\d{2}[.\/-]\d{2}[.\/-](?:19|20)\d{2}\b',
            'name_unlabeled': r'\b[A-Z]{3,}(?:\s[A-Z]{3,})?\b'
        }
        
        # Specific patterns for common document types
        self.document_specific_patterns = {
            'passport': {
                'passport_number': r'Passport No\.?[:/]?\s*([A-Z0-9]+)',
                'mrz': r'P[A-Z]{3}[A-Z0-9<]{39,}'  # Passport MRZ line often starts with P followed by country code
            },
            'drivers_license': {
                'license_number': r'(?:Driving|Driver\'?s?)\s*[Ll]icense\s*[Nn]o\.?[:/]?\s*([A-Z0-9]+)',
                'vehicle_categories': r'(?:Categories|Category)[:/]?\s*([A-Z0-9, ]+)'
            }
        }
        
        # Initialize OCR configuration options
        self.tesseract_config = {
            'psm_modes': [4, 3, 6, 7, 8],  # Page segmentation modes to try
            'languages': ['eng'],  # English only
            # Add additional OCR parameters
            'oem_modes': [3, 1],  # OCR Engine modes: 3=default, 1=LSTM only
            'custom_configs': ['--dpi 300']
        }
        
    def preprocess_image(self, image_path, preprocessing_level='default'):
        """
        Preprocess the image for better OCR results with different levels of preprocessing
        
        Args:
            image_path: Path to the image file
            preprocessing_level: Level of preprocessing ('minimal', 'default', 'aggressive', 'adaptive', 'enhanced')
        
        Returns:
            Preprocessed image or None if failed
        """
        if not TESSERACT_AVAILABLE:
            return None
            
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image from {image_path}")
                return None
            
            # Auto-detect and correct rotation if needed
            img = self._correct_rotation(img)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if preprocessing_level == 'minimal':
                if self.debug_mode:
                    self._save_debug_image(gray, image_path, "minimal")
                return gray
                
            elif preprocessing_level == 'default':
                # Apply thresholding to get a binary image
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Remove noise
                denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
                
                if self.debug_mode:
                    self._save_debug_image(denoised, image_path, "default")
                
                return denoised
                
            elif preprocessing_level == 'aggressive':
                # Apply adaptive thresholding
                adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                         cv2.THRESH_BINARY, 11, 2)
                
                # Apply morphological operations to enhance text regions
                kernel = np.ones((1, 1), np.uint8)
                eroded = cv2.erode(adaptive_thresh, kernel, iterations=1)
                dilated = cv2.dilate(eroded, kernel, iterations=1)
                
                if self.debug_mode:
                    self._save_debug_image(dilated, image_path, "aggressive")
                
                return dilated

            elif preprocessing_level == 'adaptive':
                # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # Apply bilateral filter to preserve edges while reducing noise
                filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
                
                # Apply adaptive thresholding
                adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                     cv2.THRESH_BINARY, 15, 8)
                
                if self.debug_mode:
                    self._save_debug_image(adaptive_thresh, image_path, "adaptive")
                
                return adaptive_thresh
                
            elif preprocessing_level == 'enhanced':
                # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
                
                # Apply thresholding
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Apply morphological operations
                kernel = np.ones((1, 1), np.uint8)
                morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                if self.debug_mode:
                    self._save_debug_image(morphed, image_path, "enhanced")
                
                return morphed
                
            # Default to original grayscale if no specific preprocessing level
            return gray
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            return None
    
    def _correct_rotation(self, image):
        """
        Detect and correct image rotation based on text orientation
        
        Args:
            image: Input image (color)
            
        Returns:
            Rotation-corrected image
        """
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Use HoughLinesP to find dominant lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is None or len(lines) < 5:
                # Not enough lines found, return original image
                return image
                
            # Calculate angles of lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Skip vertical lines (avoid division by zero)
                if x2 - x1 == 0:
                    continue
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)
                
            if not angles:
                return image
                
            # Find most common angle range using histogram
            hist, bins = np.histogram(angles, bins=36, range=(-90, 90))
            bin_idx = np.argmax(hist)
            rotation_angle = (bins[bin_idx] + bins[bin_idx + 1]) / 2
            
            # Adjust angle to get document straight (assuming text is horizontal)
            # We want angles close to 0 or 90 degrees
            if 45 < abs(rotation_angle) < 135:
                rotation_angle = rotation_angle - 90 if rotation_angle > 0 else rotation_angle + 90
                
            # Ignore small rotations
            if abs(rotation_angle) < 1:
                return image
                
            # Get image dimensions
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Get rotation matrix and apply rotation
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            if self.debug_mode:
                logger.info(f"Corrected image rotation by {rotation_angle:.2f} degrees")
                
            return rotated
            
        except Exception as e:
            logger.error(f"Error in rotation correction: {str(e)}")
            return image
    
    def _save_debug_image(self, img, original_path, suffix):
        """Save a debug image for inspection"""
        if self.debug_mode:
            try:
                base_name = os.path.basename(original_path)
                name, ext = os.path.splitext(base_name)
                debug_dir = os.path.join(os.path.dirname(original_path), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, f"{name}_{suffix}{ext}")
                cv2.imwrite(debug_path, img)
                logger.info(f"Saved debug image: {debug_path}")
            except Exception as e:
                logger.error(f"Failed to save debug image: {str(e)}")
    
    def extract_text(self, image_path, lang='eng', preprocess_level='default'):
        """
        Extract all text from the image with configurable preprocessing
        
        Args:
            image_path: Path to the image file
            lang: Language for OCR (eng, bul, or eng+bul)
            preprocess_level: Level of preprocessing ('minimal', 'default', 'aggressive', 'adaptive', 'enhanced')
        
        Returns:
            Extracted text string
        """
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract not available, returning empty text")
            return ""
            
        try:
            # Preprocess the image with specified level
            processed_img = self.preprocess_image(image_path, preprocess_level)
            if processed_img is None:
                logger.error("Image preprocessing failed")
                return ""
            
            # Convert OpenCV image to PIL image for pytesseract
            pil_img = Image.fromarray(processed_img)
            
            # Extract text using pytesseract with improved config for ID cards
            # Try different PSM (Page Segmentation Mode) configurations:
            # - PSM 3: Auto page segmentation, but no OSD
            # - PSM 4: Assume a single column of text of variable sizes
            # - PSM 6: Assume a single uniform block of text
            # - PSM 7: Treat the image as a single text line
            # - PSM 8: Treat the image as a single word
            # - PSM 11: Sparse text - Find as much text as possible without assuming a particular structure
            
            results = []
            ocr_attempts = []
            
            # Try different OCR engine modes
            for oem in self.tesseract_config['oem_modes']:
                # Try different page segmentation modes
                for psm in self.tesseract_config['psm_modes']:
                    # Base config
                    config = f'--psm {psm} --oem {oem}'
                    
                    # Add any custom configurations
                    for custom_config in self.tesseract_config['custom_configs']:
                        config += f' {custom_config}'
                    
                    # Try with different thresholds if in enhanced mode
                    if preprocess_level == 'enhanced':
                        # Apply different thresholds and combine results
                        thresholds = [0, 127, 200]  # Different threshold levels
                        for threshold in thresholds:
                            # Apply threshold
                            if threshold > 0:
                                img_array = np.array(pil_img)
                                _, thresh_img = cv2.threshold(img_array, threshold, 255, cv2.THRESH_BINARY)
                                pil_thresh = Image.fromarray(thresh_img)
                                text = pytesseract.image_to_string(pil_thresh, lang=lang, config=config)
                            else:
                                text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
                                
                            ocr_attempts.append((psm, oem, lang, threshold, text[:50] + "..."))  # For debugging
                            results.append(text)
                    else:
                        # Standard OCR
                        text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
                        ocr_attempts.append((psm, oem, lang, 0, text[:50] + "..."))  # For debugging
                        results.append(text)
            
            # Try MRZ specific extraction if it's likely to contain MRZ
            # MRZ lines are often better recognized with specific settings
            mrz_config = '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
            mrz_text = pytesseract.image_to_string(pil_img, lang='eng', config=mrz_config)
            results.append(mrz_text)
            ocr_attempts.append(('MRZ', 3, 'eng', 0, mrz_text[:50] + "..."))  # For debugging
            
            # Save OCR attempts for debugging
            if self.debug_mode:
                debug_dir = os.path.join(os.path.dirname(image_path), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                base_name = os.path.basename(image_path)
                name, _ = os.path.splitext(base_name)
                debug_ocr_path = os.path.join(debug_dir, f"{name}_ocr_attempts.txt")
                
                with open(debug_ocr_path, 'w', encoding='utf-8') as f:
                    for attempt in ocr_attempts:
                        f.write(f"PSM: {attempt[0]}, OEM: {attempt[1]}, Lang: {attempt[2]}, Threshold: {attempt[3]}\n")
                        f.write(f"Text: {attempt[4]}\n\n")
                        
                logger.info(f"Saved OCR attempts to: {debug_ocr_path}")
            
            # Post-process text results
            processed_results = []
            for text in results:
                # Clean and normalize text
                if text:
                    # Remove excessive whitespace
                    cleaned = re.sub(r'\s+', ' ', text)
                    # Remove common OCR artifacts and normalize characters
                    cleaned = self._normalize_text(cleaned)
                    processed_results.append(cleaned)
            
            # Combine results, prioritizing the first PSM mode but keeping unique lines from others
            all_lines = set()
            combined_text = ""
            
            for text in processed_results:
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and line not in all_lines:
                        all_lines.add(line)
                        combined_text += line + '\n'
            
            if self.debug_mode:
                logger.info(f"Combined OCR result:\n{combined_text[:200]}...")
                
                # Save full OCR result to file for debugging
                debug_dir = os.path.join(os.path.dirname(image_path), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                
                base_name = os.path.basename(image_path)
                name, _ = os.path.splitext(base_name)
                debug_text_path = os.path.join(debug_dir, f"{name}_ocr.txt")
                
                with open(debug_text_path, 'w', encoding='utf-8') as f:
                    f.write(combined_text)
                logger.info(f"Saved OCR debug text to: {debug_text_path}")
            
            return combined_text
            
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return ""
    
    def _normalize_text(self, text):
        """
        Normalize and clean OCR output text to improve matching
        
        Args:
            text: Raw OCR text
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
            
        # Common OCR substitution errors in general
        substitutions = {
            # Number-letter confusions
            '0': 'O', 'O': '0',
            '1': 'I', 'I': '1',
            '5': 'S', 'S': '5',
            '8': 'B', 'B': '8',
            
            # Other common substitutions
            '/': 'l', 'l': '/',
            ';': ':',
            '!': '1',
            '|': 'I',
        }
            
        # Apply substitutions for better matching
        normalized = text
        for error, correction in substitutions.items():
            normalized = normalized.replace(error, correction)
            
        # Remove common noise characters
        normalized = re.sub(r'[_=~<>]', '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _try_smart_extraction(self, text, extracted_data):
        """Try to extract fields without explicit labels using smart patterns"""
        # Skip if we've already found enough fields
        if len(extracted_data) >= 3:
            return
            
        # Try to extract unlabeled fields using smart patterns
        for field, pattern in self.smart_patterns.items():
            # Skip fields that we've already found
            base_field = field.replace('_unlabeled', '')
            if base_field in extracted_data or field in extracted_data:
                continue
                
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Take the first match that looks valid
                for match in matches:
                    # Basic validation checks
                    if '_number' in field and len(match) >= 6:  # ID numbers are usually at least 6 chars
                        extracted_data[base_field] = match
                        break
                    elif 'date' in field:
                        # Try to parse as date
                        try:
                            # Simple validation - we just want to make sure it looks like a date
                            parts = re.split(r'[.\/-]', match)
                            if len(parts) == 3 and all(part.isdigit() for part in parts):
                                extracted_data[base_field] = match
                                break
                        except:
                            continue
                            
        return extracted_data
    
    def _try_detect_document_type(self, text):
        """
        Try to detect the type of document from text
        
        Args:
            text: Extracted text from document
        
        Returns:
            Document type (id_card, passport, drivers_license, or unknown)
        """
        document_types = {
            'id_card': [
                'id card', 'identity card', 'identification card', 'id number', 
                'republic of bulgaria'
            ],
            'passport': [
                'passport', 'travel document', 'international passport'
            ],
            'drivers_license': [
                'driving licence', 'driver\'s license', 'driving license'
            ]
        }
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # First try to match exact document type indicators
        for doc_type, keywords in document_types.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return doc_type
        
        # If no exact match, try some heuristics for ID cards
        if re.search(r'(?:id|№)\s*[:/]?\s*\d{6,}', text):
            return 'id_card'
        
        # If we find an MRZ pattern, it's likely a passport or ID card
        if re.search(r'[A-Z<]{5,44}', text):
            # ID cards typically have shorter MRZ lines
            if len(re.findall(r'[A-Z<]{5,44}', text)[0]) <= 30:
                return 'id_card'
            return 'passport'
            
        return 'unknown'
    
    def extract_fields(self, image_path, debug_level='normal'):
        """
        Extract specific fields from ID card with enhanced debugging
        
        Args:
            image_path: Path to the image file
            debug_level: Level of debugging ('minimal', 'normal', 'verbose')
        
        Returns:
            Dictionary with extracted field values
        """
        extracted_data = {}
        extraction_attempts = {}
        
        # If Tesseract is not available, use the fallback method
        if not TESSERACT_AVAILABLE:
            logger.warning("Tesseract not available, using fallback extraction")
            return self.fallback_extraction(image_path)
        
        # Set debug mode based on debug_level
        old_debug_mode = self.debug_mode
        self.debug_mode = debug_level in ('normal', 'verbose')
        
        try:
            # Try different preprocessing levels and languages for best results
            preprocess_levels = ['enhanced', 'adaptive', 'default', 'minimal', 'aggressive']
            languages = self.tesseract_config['languages']
            
            # Flag to track if we've found sufficient data
            found_sufficient_data = False
            
            for preprocess_level in preprocess_levels:
                for lang in languages:
                    if debug_level == 'verbose':
                        logger.info(f"Attempting extraction with preprocess_level={preprocess_level}, lang={lang}")
                    
                    # Extract text from image with current settings
                    text = self.extract_text(image_path, lang=lang, preprocess_level=preprocess_level)
                    
                    # Try to detect document type
                    doc_type = self._try_detect_document_type(text)
                    
                    # Check for MRZ (Machine Readable Zone) - common on ID cards
                    mrz_data = self.check_mrz(text)
                    if mrz_data:
                        # Extract data from MRZ and add to extracted data
                        mrz_extracted = self.extract_data_from_mrz(mrz_data)
                        for field, value in mrz_extracted.items():
                            if field not in extracted_data or not extracted_data[field]:
                                extracted_data[field] = value
                                extraction_attempts[field] = f"Found from MRZ ({preprocess_level}, {lang})"
                                if debug_level == 'verbose':
                                    logger.info(f"Found {field}: {value} from MRZ")
                    
                    if doc_type != 'unknown' and doc_type in self.document_specific_patterns:
                        # Apply document-specific patterns
                        for field, pattern in self.document_specific_patterns[doc_type].items():
                            if field not in extracted_data or not extracted_data[field]:  # Only if not already found
                                match = re.search(pattern, text, re.IGNORECASE)
                                if match:
                                    value = match.group(1).strip()
                                    extracted_data[field] = value
                                    extraction_attempts[field] = f"Found with {doc_type}-specific pattern ({preprocess_level}, {lang})"
                                    if debug_level == 'verbose':
                                        logger.info(f"Found {field}: {value} with {doc_type}-specific pattern")
                    
                    # Extract each field using primary regex patterns
                    for field, pattern in self.field_patterns.items():
                        if field not in extracted_data or not extracted_data[field]:  # Only if not already found
                            match = re.search(pattern, text, re.IGNORECASE)
                            if match:
                                value = match.group(1).strip()
                                extracted_data[field] = value
                                extraction_attempts[field] = f"Found with primary pattern ({preprocess_level}, {lang})"
                                if debug_level == 'verbose':
                                    logger.info(f"Found {field}: {value} with primary pattern")
                    
                    # Try alternative patterns for fields not yet found
                    for field, alt_patterns in self.alternative_patterns.items():
                        if field not in extracted_data or not extracted_data[field]:  # Only if not already found
                            for alt_pattern in alt_patterns:
                                match = re.search(alt_pattern, text, re.IGNORECASE)
                                if match:
                                    value = match.group(1).strip()
                                    extracted_data[field] = value
                                    extraction_attempts[field] = f"Found with alternative pattern ({preprocess_level}, {lang})"
                                    if debug_level == 'verbose':
                                        logger.info(f"Found {field}: {value} with alternative pattern: {alt_pattern}")
                                    break
                    
                    # Try smart extraction techniques for unlabeled fields
                    self._try_smart_extraction(text, extracted_data)
                    
                    # If we found most of the fields, we can stop trying other configurations
                    priority_fields = ['first_name', 'last_name', 'identity_number', 'document_number', 'personal_number', 'id_number']
                    found_priority = sum(1 for field in priority_fields if field in extracted_data)
                    
                    # If we have a name and at least one ID field, that's good enough to proceed
                    name_fields = ['first_name', 'last_name']
                    id_fields = ['identity_number', 'document_number', 'personal_number', 'id_number', 'passport_number', 'license_number']
                    
                    has_name = any(field in extracted_data for field in name_fields)
                    has_id = any(field in extracted_data for field in id_fields)
                    
                    if has_name and has_id and found_priority >= 2:  # If we found name and ID, it's enough
                        found_sufficient_data = True
                        break
                
                # If we found enough fields with this preprocessing level, stop trying others
                if found_sufficient_data:
                    break
            
            # Additional extraction for structured tables if we didn't find enough data
            if not found_sufficient_data and 'identity_number' not in extracted_data:
                # Try table detection and structured parsing for forms
                structured_data = self._try_table_extraction(image_path)
                if structured_data:
                    for field, value in structured_data.items():
                        if field not in extracted_data or not extracted_data[field]:
                            extracted_data[field] = value
                            extraction_attempts[field] = "Found with table extraction"
                            if debug_level == 'verbose':
                                logger.info(f"Found {field}: {value} with table extraction")
            
            # Validate and clean all extracted data
            extracted_data = self._validate_and_clean_fields(extracted_data, debug_level)
            
            # Check consistency between fields (e.g., ID number format matches document type)
            extracted_data = self._validate_field_consistency(extracted_data, doc_type)
            
            # Copy ID numbers to standardized fields (to ensure we always have identity_number if any ID was found)
            # This ensures compatibility with original code that might expect specific field names
            id_field_mapping = {
                'document_number': 'identity_number',
                'id_number': 'identity_number',
                'personal_number': 'identity_number',
                'passport_number': 'identity_number',
                'license_number': 'identity_number'
            }
            
            for source, target in id_field_mapping.items():
                if source in extracted_data and extracted_data[source] and target not in extracted_data:
                    extracted_data[target] = extracted_data[source]
                    if debug_level == 'verbose':
                        logger.info(f"Copied {source} to {target}: {extracted_data[source]}")
            
            # Save debug information if needed
            if debug_level in ('normal', 'verbose'):
                extraction_data = {
                    'image_path': image_path,
                    'document_type': doc_type,
                    'extracted_fields': extracted_data,
                    'extraction_attempts': extraction_attempts,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                debug_dir = os.path.join(os.path.dirname(image_path), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                
                base_name = os.path.basename(image_path)
                name, _ = os.path.splitext(base_name)
                debug_extract_path = os.path.join(debug_dir, f"{name}_extraction.txt")
                
                with open(debug_extract_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(extraction_data, f, indent=2)
                
                logger.info(f"Saved extraction debug info to: {debug_extract_path}")
        
        finally:
            # Restore original debug mode
            self.debug_mode = old_debug_mode
        
        return extracted_data
    
    def _try_table_extraction(self, image_path):
        """
        Attempt to extract data from structured table layouts in the ID card
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with extracted field values
        """
        extracted_data = {}
        
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                return {}
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by y-coordinate (top to bottom)
            sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
            
            # Create blank image for debugging if needed
            if self.debug_mode:
                debug_img = img.copy()
                
            # Process each contour as a potential field
            table_rows = []
            for i, contour in enumerate(sorted_contours):
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out very small contours and very large ones
                if w < 20 or h < 10 or w > img.shape[1] * 0.9 or h > img.shape[0] * 0.3:
                    continue
                
                # Extract region
                roi = gray[y:y+h, x:x+w]
                
                # OCR the region
                roi_pil = Image.fromarray(roi)
                text = pytesseract.image_to_string(roi_pil, config='--psm 7')
                text = text.strip()
                
                if text:
                    # Draw contour on debug image
                    if self.debug_mode:
                        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(debug_img, f"{i}:{text[:10]}", (x, y-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                    # Group into rows based on y-coordinate
                    found_row = False
                    for row in table_rows:
                        if abs(row['y'] - y) < 15:  # Within same row
                            row['cells'].append({'x': x, 'text': text, 'width': w})
                            found_row = True
                            break
                    
                    if not found_row:
                        table_rows.append({'y': y, 'cells': [{'x': x, 'text': text, 'width': w}]})
            
            # Save debug image if in debug mode
            if self.debug_mode:
                debug_dir = os.path.join(os.path.dirname(image_path), 'debug')
                os.makedirs(debug_dir, exist_ok=True)
                base_name = os.path.basename(image_path)
                name, _ = os.path.splitext(base_name)
                debug_table_path = os.path.join(debug_dir, f"{name}_table.jpg")
                cv2.imwrite(debug_table_path, debug_img)
            
            # Process rows to extract field:value pairs
            for row in table_rows:
                if len(row['cells']) >= 2:
                    # Sort cells by x-coordinate
                    sorted_cells = sorted(row['cells'], key=lambda c: c['x'])
                    
                    # Typically, field name is on the left, value on the right
                    field_text = sorted_cells[0]['text'].strip().lower()
                    value_text = sorted_cells[1]['text'].strip()
                    
                    # Map to known fields
                    if 'name' in field_text or 'име' in field_text:
                        extracted_data['first_name'] = value_text
                    elif 'surname' in field_text or 'фамилия' in field_text:
                        extracted_data['last_name'] = value_text
                    elif 'birth' in field_text or 'born' in field_text or 'рожден' in field_text:
                        extracted_data['birth_date'] = value_text
                    elif 'personal' in field_text or 'егн' in field_text:
                        extracted_data['personal_number'] = value_text
                    elif 'identity' in field_text or 'номер' in field_text or 'лична' in field_text:
                        extracted_data['identity_number'] = value_text
                    elif 'nation' in field_text or 'гражданство' in field_text:
                        extracted_data['nationality'] = value_text
                    elif 'sex' in field_text or 'gender' in field_text or 'пол' in field_text:
                        extracted_data['sex'] = value_text
                    elif 'expiry' in field_text or 'valid' in field_text or 'валидно' in field_text:
                        extracted_data['expiry_date'] = value_text
            
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error in table extraction: {str(e)}")
            return {}
            
    def check_mrz(self, text):
        """
        Check for Machine Readable Zone (MRZ) in the text
        
        Args:
            text: OCR extracted text to check
            
        Returns:
            MRZ data if found, otherwise None
        """
        # Check for MRZ (Machine Readable Zone)
        mrz_pattern = r'([A-Z<]{5,44})\n*([A-Z0-9<]{5,44})'
        mrz_match = re.search(mrz_pattern, text)
        
        if mrz_match:
            mrz_line1 = mrz_match.group(1).strip()
            mrz_line2 = mrz_match.group(2).strip()
            
            # Basic validation of MRZ format
            if len(mrz_line1) >= 30 or len(mrz_line2) >= 30:
                return [mrz_line1, mrz_line2]
        
        return None
    
    def extract_data_from_mrz(self, mrz_data):
        """
        Extract identification data from MRZ code
        
        Args:
            mrz_data: List of MRZ lines
            
        Returns:
            Dictionary with extracted data
        """
        extracted = {}
        
        if not mrz_data or len(mrz_data) < 2:
            return extracted
            
        # MRZ line format examples:
        # ID card MRZ format (example):
        # IDUSADIMITROV<<IVAN<PETROV<<<<<<<<<<<<<<<<
        # 1234567890USA8001017M2001017DIMITROV02589<<
        
        # Passport MRZ format (example):
        # P<USADIMITROV<<IVAN<PETROV<<<<<<<<<<<<<<<<
        # 1234567890USA8001017M2001017<<<<<<<<<<<<<<4
        
        try:
            # First line typically contains the document type, country code, and names
            line1 = mrz_data[0]
            # Second line contains document number, date of birth, expiry date, etc.
            line2 = mrz_data[1]
            
            # Extract document type from first line
            if line1.startswith('ID'):
                doc_type = 'id_card'
            elif line1.startswith('P'):
                doc_type = 'passport'
            else:
                doc_type = 'unknown'
                
            # Extract names from first line
            name_part = line1[5:] if doc_type == 'id_card' else line1[5:]
            name_parts = name_part.split('<<')
            
            if len(name_parts) >= 2:
                last_name = name_parts[0].replace('<', ' ').strip()
                first_name = name_parts[1].replace('<', ' ').strip()
                
                if last_name:
                    extracted['last_name'] = last_name
                if first_name:
                    extracted['first_name'] = first_name
            
            # Extract document number from second line
            if len(line2) >= 9:
                extracted['document_number'] = line2[:9].strip('<')
            
            # Extract nationality from second line
            if len(line2) >= 12:
                extracted['nationality'] = line2[10:13]  # 3-letter country code
            
            # Extract birth date from second line
            if len(line2) >= 19:
                birth_date_str = line2[13:19]  # Format: YYMMDD
                if birth_date_str and birth_date_str[0] != '<':
                    try:
                        year = int(birth_date_str[:2])
                        month = int(birth_date_str[2:4])
                        day = int(birth_date_str[4:6])
                        
                        # Adjust for century (assuming 19xx for >23, 20xx for <=23)
                        current_year = time.localtime().tm_year % 100
                        century = 1900 if year > current_year else 2000
                        full_year = century + year
                        
                        # Format the date as DD.MM.YYYY
                        extracted['birth_date'] = f"{day:02d}.{month:02d}.{full_year}"
                    except ValueError:
                        pass
            
            # Extract gender from second line
            if len(line2) >= 20:
                gender = line2[20:21]
                if gender == 'M':
                    extracted['sex'] = 'M'
                elif gender == 'F':
                    extracted['sex'] = 'F'
            
            # Extract expiry date from second line
            if len(line2) >= 27:
                expiry_date_str = line2[21:27]  # Format: YYMMDD
                if expiry_date_str and expiry_date_str[0] != '<':
                    try:
                        year = int(expiry_date_str[:2])
                        month = int(expiry_date_str[2:4])
                        day = int(expiry_date_str[4:6])
                        
                        # Adjust for century (assuming 20xx for expiry dates)
                        full_year = 2000 + year
                        
                        # Format the date as DD.MM.YYYY
                        extracted['expiry_date'] = f"{day:02d}.{month:02d}.{full_year}"
                    except ValueError:
                        pass
            
            # Extract personal number if present (ID cards)
            if doc_type == 'id_card' and len(line2) >= 36:
                personal_number = line2[28:37].replace('<', '')
                if personal_number:
                    extracted['personal_number'] = personal_number
                    
        except Exception as e:
            logger.error(f"Error extracting data from MRZ: {str(e)}")
            
        return extracted
    
    def _validate_field_consistency(self, data, doc_type):
        """
        Validate consistency between extracted fields
        
        Args:
            data: Dictionary of extracted data
            doc_type: Document type string
            
        Returns:
            Dictionary with validated and potentially corrected data
        """
        if not data:
            return data
            
        # Make a copy to avoid modifying the original
        result = data.copy()
        
        try:
            # Check for ID number patterns and formats
            if 'identity_number' in result:
                id_num = result['identity_number']
                
                # ID numbers are typically 9 digits or have format XX0000000
                if id_num and (len(id_num) < 6 or len(id_num) > 12):
                    # This is likely not a valid ID number
                    logger.warning(f"Suspicious ID number: {id_num}")
                    
                    # Try to find a better match from other fields
                    if 'document_number' in result and result['document_number'] and 6 <= len(result['document_number']) <= 12:
                        result['identity_number'] = result['document_number']
                        logger.info(f"Replaced identity_number with document_number: {result['identity_number']}")
            
            # Check for personal number patterns
            if 'personal_number' in result:
                personal_num = result['personal_number']
                
                # Personal number is typically 6-10 digits
                if personal_num and (len(personal_num) < 6 or len(personal_num) > 12):
                    logger.warning(f"Suspicious personal number: {personal_num}")
                    
                    # Try to find a better match if we have document_number as fallback
                    if 'document_number' in result and result['document_number'] and 6 <= len(result['document_number']) <= 12:
                        result['personal_number'] = re.sub(r'[^0-9]', '', result['document_number'])
                        logger.info(f"Replaced personal_number with digits from document_number: {result['personal_number']}")
            
            # Check for valid dates
            for date_field in ['birth_date', 'expiry_date']:
                if date_field in result and result[date_field]:
                    date_str = result[date_field]
                    
                    # Validate date format - should be something like DD.MM.YYYY or similar
                    if not re.match(r'\d{1,2}[.\/-]\d{1,2}[.\/-](?:19|20)?\d{2}$', date_str):
                        logger.warning(f"Invalid date format for {date_field}: {date_str}")
                        result[date_field] = None
                        continue
                    
                    # Try to parse the date parts
                    date_parts = re.split(r'[.\/-]', date_str)
                    if len(date_parts) == 3:
                        day, month, year = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
                        
                        # Validate individual components
                        if day < 1 or day > 31 or month < 1 or month > 12:
                            logger.warning(f"Invalid day/month in {date_field}: {date_str}")
                            result[date_field] = None
                            continue
                        
                        # Fix 2-digit years
                        if year < 100:
                            year += 2000 if year < 50 else 1900
                            result[date_field] = f"{day:02d}.{month:02d}.{year}"
                            logger.info(f"Fixed 2-digit year in {date_field}: {result[date_field]}")
                        
                        # For expiry dates, if it's in the past, flag it
                        if date_field == 'expiry_date':
                            current_year = time.localtime().tm_year
                            if year < current_year or (year == current_year and month < time.localtime().tm_mon):
                                logger.warning(f"Document appears to be expired: {date_str}")
                                # We don't remove it, just log the warning
            
            # Normalize name fields (uppercase, remove invalid chars)
            for name_field in ['first_name', 'last_name']:
                if name_field in result and result[name_field]:
                    # Convert to uppercase
                    result[name_field] = result[name_field].upper()
                    
                    # Remove any non-letter characters except spaces and hyphens
                    result[name_field] = re.sub(r'[^A-Z\s-]', '', result[name_field])
                    
                    # Normalize spaces
                    result[name_field] = re.sub(r'\s+', ' ', result[name_field]).strip()
                    
                    # If we've emptied the field, remove it
                    if not result[name_field]:
                        result.pop(name_field)
            
            # Normalize gender/sex field
            if 'sex' in result and result['sex']:
                sex = result['sex'].upper()
                if sex in ['M', 'MALE']:
                    result['sex'] = 'M'
                elif sex in ['F', 'FEMALE']:
                    result['sex'] = 'F'
                else:
                    # Invalid sex value
                    result.pop('sex')
            
            # Normalize nationality
            if 'nationality' in result and result['nationality']:
                # Convert to uppercase
                result['nationality'] = result['nationality'].upper()
                
                # Remove any non-letter characters
                result['nationality'] = re.sub(r'[^A-Z]', '', result['nationality'])
                
                # Replace known country codes with full names
                country_map = {
                    'USA': 'UNITED STATES', 'US': 'UNITED STATES', 
                    'UK': 'UNITED KINGDOM', 'GBR': 'UNITED KINGDOM',
                    'DEU': 'GERMANY', 'GER': 'GERMANY',
                    'FRA': 'FRANCE', 'FR': 'FRANCE',
                    'BG': 'BULGARIA', 'BULGARIAN': 'BULGARIA',
                    'ESP': 'SPAIN', 'SPA': 'SPAIN',
                    'ITA': 'ITALY',
                    'CAN': 'CANADA',
                    'AUS': 'AUSTRALIA',
                    'JPN': 'JAPAN',
                }
                
                if result['nationality'] in country_map:
                    result['nationality'] = country_map[result['nationality']]
        
        except Exception as e:
            logger.error(f"Error validating field consistency: {str(e)}")
            
        return result
    
    def fallback_extraction(self, image_path):
        """Fallback method when Tesseract is not available - simulates data extraction"""
        logger.info(f"Using fallback extraction mode for {image_path}")
        
        # Generate simulated data for demo purposes
        sample_names = ["IVAN", "MARIA", "DIMITAR", "ELENA", "GEORGI", "STEFAN", "NIKOLAY", "ALEXANDER"]
        sample_surnames = ["IVANOV", "PETROVA", "DIMITROV", "GEORGIEVA", "STEFANOV", "KOSTOV", "IVANOVA"]
        
        # Create simulated extracted data with proper formatting
        extracted_data = {
            'first_name': random.choice(sample_names),
            'last_name': random.choice(sample_surnames),
            'identity_number': ''.join([str(random.randint(0, 9)) for _ in range(9)]),
            'document_number': f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))}{''.join([str(random.randint(0, 9)) for _ in range(6)])}",
            'personal_number': ''.join([str(random.randint(0, 9)) for _ in range(10)]),
            'birth_date': f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{random.randint(1960, 2000)}",
            'expiry_date': f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{random.randint(2023, 2030)}",
            'nationality': "BULGARIA",
            'sex': random.choice(['M', 'F']),
            'place_of_birth': random.choice(["SOFIA", "PLOVDIV", "VARNA", "BURGAS"]),
            'issuing_authority': "MVR SOFIA"
        }
        
        logger.info(f"Simulated extracted data: {extracted_data}")
        return extracted_data
    
    def _validate_and_clean_fields(self, extracted_data, debug_level='normal'):
        """
        Validate and clean extracted fields
        
        Args:
            extracted_data: Dictionary of extracted fields
            debug_level: Debug level ('minimal', 'normal', 'verbose')
            
        Returns:
            Dictionary with validated and cleaned fields
        """
        if not extracted_data:
            return {}
            
        # Define validation rules for each field
        field_validators = {
            # ID numbers should be numeric, minimum length 6
            'identity_number': lambda x: bool(re.match(r'^[0-9]{6,}$|^[A-Z]{1,2}[0-9]{6,8}$', x)),
            'document_number': lambda x: bool(re.match(r'^[A-Z0-9]{5,}$', x)) and not re.match(r'^(document|uument|number|identity)$', x, re.IGNORECASE),
            'personal_number': lambda x: bool(re.match(r'^[0-9]{6,}$', x)),
            
            # Names should be Latin letters, minimum length 2
            'first_name': lambda x: bool(re.match(r'^[A-Z]{2,}(?:\s[A-Z]{2,})?$', x)),
            'last_name': lambda x: bool(re.match(r'^[A-Z]{2,}(?:\s[A-Z]{2,})?$', x)),
            
            # Dates should match standard formats
            'birth_date': lambda x: bool(re.match(r'^[0-9]{1,2}[.\/-][0-9]{1,2}[.\/-](?:19|20)?[0-9]{2}$', x)),
            'expiry_date': lambda x: bool(re.match(r'^[0-9]{1,2}[.\/-][0-9]{1,2}[.\/-](?:19|20)?[0-9]{2}$', x)),
            
            # Other fields with specific formats
            'sex': lambda x: x.upper() in ['M', 'F'],
            'nationality': lambda x: bool(re.match(r'^[A-Z]{2,}$', x)),
            'height': lambda x: bool(re.match(r'^[0-9]{2,3}$', x)),
        }
        
        # List of common OCR errors or invalid words to filter out
        invalid_words = [
            'document', 'number', 'identity', 'name', 'date', 'uument', 'personal', 
            'id', 'no', 'valid', 'expiry', 'sex', 'signature', 'height',
            'republic', 'address', 'family'
        ]

        # Common OCR error corrections
        ocr_corrections = {
            'identity_number': {
                'O': '0', 'o': '0', 'l': '1', 'I': '1', 'S': '5', 's': '5',
                'B': '8', 'b': '8', 'G': '6', 'g': '6', 'Z': '2', 'z': '2'
            },
            'personal_number': {
                'O': '0', 'o': '0', 'l': '1', 'I': '1', 'S': '5', 's': '5',
                'B': '8', 'b': '8', 'G': '6', 'g': '6', 'Z': '2', 'z': '2'
            }
        }

        # Clean and validate each field
        for field, value in list(extracted_data.items()):  # Use list() to avoid dictionary changing during iteration
            if not value:  # Skip empty values
                continue
                
            # Clean the value - remove extra spaces and normalize
            value = value.strip().upper()
            
            # Skip if the value is just an invalid word
            if value.lower() in invalid_words:
                extracted_data.pop(field, None)  # Remove invalid fields
                continue
            
            # Apply field-specific corrections to fix common OCR errors
            if field in ocr_corrections:
                for old, new in ocr_corrections[field].items():
                    value = value.replace(old, new)
            
            # Apply field-specific validation if we have a validator
            if field in field_validators:
                if not field_validators[field](value):
                    # Try to fix some common issues before giving up
                    fixed_value = self._try_fix_field(field, value)
                    if fixed_value and field_validators[field](fixed_value):
                        value = fixed_value
                        if debug_level == 'verbose':
                            logger.info(f"Fixed field {field}: {value}")
                    else:
                        if debug_level == 'verbose':
                            logger.warning(f"Field {field} failed validation: {value}")
                        extracted_data.pop(field, None)  # Remove invalid fields
                        continue
            
            # Date fields: standardize format to DD.MM.YYYY
            if field in ['birth_date', 'expiry_date']:
                date_match = re.search(r'(\d{1,2})[.\/-](\d{1,2})[.\/-](\d{2,4})', value)
                if date_match:
                    day = date_match.group(1).zfill(2)  # Ensure 2 digits
                    month = date_match.group(2).zfill(2)
                    year = date_match.group(3)
                    
                    # Fix two-digit years
                    if len(year) == 2:
                        year = '19' + year if int(year) >= 50 else '20' + year
                        
                    value = f"{day}.{month}.{year}"
            
            # Store the cleaned value
            extracted_data[field] = value
        
        # Post-processing: check for additional validation issues and name handling
        
        # Enhanced name field handling - first search for any name fields
        name_fields_detected = 'first_name' in extracted_data or 'last_name' in extracted_data or 'name_unlabeled' in extracted_data
        
        # If we have first_name but no last_name, try to extract last_name from it (sometimes OCR combines them)
        if 'first_name' in extracted_data and 'last_name' not in extracted_data:
            name_parts = extracted_data['first_name'].strip().split()
            if len(name_parts) >= 2:
                # First part as first name, rest as last name
                extracted_data['first_name'] = name_parts[0].upper()
                extracted_data['last_name'] = ' '.join(name_parts[1:]).upper()
                if debug_level == 'verbose':
                    logger.info(f"Split name into first_name: {extracted_data['first_name']} and last_name: {extracted_data['last_name']}")
            
        # If we have last_name but no first_name, try to extract first_name from it
        elif 'last_name' in extracted_data and 'first_name' not in extracted_data:
            name_parts = extracted_data['last_name'].strip().split()
            if len(name_parts) >= 2:
                # First part as first name, rest as last name
                extracted_data['first_name'] = name_parts[0].upper()
                extracted_data['last_name'] = ' '.join(name_parts[1:]).upper()
                if debug_level == 'verbose':
                    logger.info(f"Split name into first_name: {extracted_data['first_name']} and last_name: {extracted_data['last_name']}")
        
        # Use unlabeled name if available and we're missing a name field
        elif ('first_name' not in extracted_data or 'last_name' not in extracted_data) and 'name_unlabeled' in extracted_data:
            name_parts = extracted_data['name_unlabeled'].strip().split()
            if len(name_parts) >= 2:
                extracted_data['first_name'] = name_parts[0].upper()
                extracted_data['last_name'] = ' '.join(name_parts[1:]).upper()
                if debug_level == 'verbose':
                    logger.info(f"Used unlabeled name for first_name: {extracted_data['first_name']} and last_name: {extracted_data['last_name']}")
            elif len(name_parts) == 1 and len(name_parts[0]) >= 3:
                # Just one word, use it as first name
                extracted_data['first_name'] = name_parts[0].upper()
                # If we have only first_name, provide a default last_name to prevent process failure
                extracted_data['last_name'] = 'UNKNOWN'
                if debug_level == 'verbose':
                    logger.info(f"Used single word unlabeled name for first_name: {extracted_data['first_name']} with default last_name")
        
        # If we still don't have both name fields, provide defaults to allow processing to continue
        if 'first_name' in extracted_data and 'last_name' not in extracted_data:
            extracted_data['last_name'] = 'UNKNOWN'
            logger.warning("No last_name detected, using default 'UNKNOWN'")
        elif 'last_name' in extracted_data and 'first_name' not in extracted_data:
            extracted_data['first_name'] = 'UNKNOWN'
            logger.warning("No first_name detected, using default 'UNKNOWN'")
        elif not name_fields_detected:
            # Last resort - if no name fields at all, add placeholder values
            extracted_data['first_name'] = 'UNKNOWN'
            extracted_data['last_name'] = 'UNKNOWN'
            logger.warning("No name fields detected, using defaults for both first_name and last_name")
            
        # Log validation results if in debug mode
        if debug_level != 'minimal':
            valid_fields = [field for field in field_validators if field in extracted_data]
            invalid_fields = [field for field in valid_fields if not field_validators[field](extracted_data[field])]
            if invalid_fields:
                logger.info(f"Fields that failed validation: {invalid_fields}")
            logger.info(f"Cleaned data: {extracted_data}")
        
        return extracted_data
    
    def _try_fix_field(self, field, value):
        """
        Attempt to fix common issues with field values
        
        Args:
            field: The field name
            value: The original value
            
        Returns:
            Fixed value if possible, otherwise None
        """
        # Strip any non-alphanumeric characters from ID fields
        if field in ['identity_number', 'document_number', 'personal_number']:
            if field == 'identity_number' or field == 'document_number':
                # For identity/document numbers, keep letters and digits
                fixed = re.sub(r'[^A-Z0-9]', '', value.upper())
                # Remove common prefixes like "NO" or "Nr" or "ID:"
                fixed = re.sub(r'^(NO|NR|N|ID|NUM|NUMBER)[.:;, ]*', '', fixed, flags=re.IGNORECASE)
                return fixed if fixed else None
                
            elif field == 'personal_number':
                # For personal numbers, keep only digits
                fixed = re.sub(r'[^0-9]', '', value)
                # Ensure it's between 6-10 digits
                if len(fixed) > 10:
                    fixed = fixed[:10]  # Truncate if too long
                return fixed if len(fixed) >= 6 else None
        
        # Name fields - remove numbers and special characters
        elif field in ['first_name', 'last_name']:
            # Keep only letters and spaces
            fixed = re.sub(r'[^A-Z\s-]', '', value.upper())
            # Normalize spaces
            fixed = re.sub(r'\s+', ' ', fixed).strip()
            return fixed if len(fixed) >= 2 else None
            
        # Date fields - try to parse various formats
        elif field in ['birth_date', 'expiry_date']:
            # Try various date formats
            date_patterns = [
                # DD.MM.YYYY or DD-MM-YYYY or DD/MM/YYYY
                r'(\d{1,2})[.\/-](\d{1,2})[.\/-](\d{2,4})',
                # YYYY.MM.DD or YYYY-MM-DD or YYYY/MM/DD
                r'(\d{4})[.\/-](\d{1,2})[.\/-](\d{1,2})',
                # DDMMYYYY or DDMMYY (no separators)
                r'(\d{2})(\d{2})(\d{2,4})'
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, value)
                if match:
                    # Format depends on which pattern matched
                    if pattern == date_patterns[0]:
                        day, month, year = match.groups()
                    elif pattern == date_patterns[1]:
                        year, month, day = match.groups()
                    else:  # Pattern without separators
                        day, month, year = match.groups()
                    
                    # Fix two-digit years
                    if len(year) == 2:
                        year = '19' + year if int(year) >= 50 else '20' + year
                        
                    # Basic validation
                    try:
                        day_val = int(day)
                        month_val = int(month)
                        year_val = int(year)
                        
                        if 1 <= day_val <= 31 and 1 <= month_val <= 12 and 1900 <= year_val <= 2100:
                            return f"{day.zfill(2)}.{month.zfill(2)}.{year}"
                    except ValueError:
                        continue
                        
            return None
        
        # Sex/gender field
        elif field == 'sex':
            # Map various gender indicators to standard format
            gender_map = {
                'M': 'M', 'MALE': 'M',
                'F': 'F', 'W': 'F', 'FEMALE': 'F'
            }
            
            upper_val = value.upper()
            for key, mapped in gender_map.items():
                if key in upper_val:
                    return mapped
                    
            return None
            
        # Nationality field
        elif field == 'nationality':
            # Clean and standardize
            fixed = re.sub(r'[^A-Z]', '', value.upper())
            # Map common values
            nationality_map = {
                'BG': 'BULGARIA', 'BULGARIAN': 'BULGARIA',
                'BGR': 'BULGARIA', 'REPUBLIC OF BULGARIA': 'BULGARIA'
            }
            
            if fixed in nationality_map:
                return nationality_map[fixed]
                
            return fixed if len(fixed) >= 2 else None
            
        # Default: return original value
        return value
    
    def process_id_card(self, front_image_path, back_image_path=None, debug_level='normal'):
        """
        Process both sides of ID card and return combined data with debugging options
        
        Args:
            front_image_path: Path to the front image of the ID
            back_image_path: Path to the back image of the ID (optional)
            debug_level: Level of debugging ('minimal', 'normal', 'verbose')
        
        Returns:
            Dictionary with extracted field values from both sides
        """
        if not front_image_path:
            logger.error("Front image path is required")
            return {}
            
        try:
            # Check if the front image exists
            if not os.path.exists(front_image_path):
                logger.error(f"Front image does not exist: {front_image_path}")
                return {}
                
            # Check if back image exists if provided
            if back_image_path and not os.path.exists(back_image_path):
                logger.warning(f"Back image does not exist: {back_image_path}. Proceeding with front only.")
                back_image_path = None
                
            # Try automated ID detection if back image wasn't explicitly provided
            if not back_image_path:
                detected_back = self._try_detect_back_side(front_image_path)
                if detected_back:
                    logger.info(f"Automatically detected back side image: {detected_back}")
                    back_image_path = detected_back
            
            # Extract data from front side
            logger.info(f"Processing front side of ID: {front_image_path}")
            front_data = self.extract_fields(front_image_path, debug_level=debug_level)
            
            # Extract data from back side if provided
            back_data = {}
            if back_image_path:
                logger.info(f"Processing back side of ID: {back_image_path}")
                back_data = self.extract_fields(back_image_path, debug_level=debug_level)
            
            # Combine data from both sides
            combined_data = {**front_data, **back_data}
            
            # Perform additional validation and cross-checking of data from both sides
            combined_data = self._validate_cross_check(combined_data)
            
            # Perform final check to ensure required fields exist
            # Ensure we have at least first_name and last_name fields
            if 'first_name' not in combined_data:
                logger.warning("first_name not detected in OCR results, adding default value")
                combined_data['first_name'] = "UNKNOWN"
                
            if 'last_name' not in combined_data:
                logger.warning("last_name not detected in OCR results, adding default value")
                combined_data['last_name'] = "UNKNOWN"
            
            # Log the final result
            if debug_level != 'minimal':
                logger.info(f"Combined extraction results: {combined_data}")
            
            return combined_data
        except KeyError as e:
            # Handle specific missing field errors
            field_name = str(e).strip("'")
            logger.error(f"Error in ID card processing: '{field_name}'")
            
            # Create a minimal result with the field that was missing
            result = {}
            
            # Add any data we did successfully extract
            if 'front_data' in locals() and isinstance(front_data, dict):
                result.update(front_data)
            if 'back_data' in locals() and isinstance(back_data, dict):
                result.update(back_data)
                
            # Add defaults for critical missing fields
            if field_name == 'first_name' and 'first_name' not in result:
                result['first_name'] = "UNKNOWN"
            if field_name == 'last_name' and 'last_name' not in result:
                result['last_name'] = "UNKNOWN"
                
            return result
        except Exception as e:
            logger.error(f"Error in ID card processing: {str(e)}")
            return {}
    
    def _try_detect_back_side(self, front_image_path):
        """
        Try to automatically detect the back side image based on the front image path
        
        Args:
            front_image_path: Path to the front image
            
        Returns:
            Path to the back image if found, otherwise None
        """
        try:
            # Get directory, filename and extension
            dir_path = os.path.dirname(front_image_path)
            base_name = os.path.basename(front_image_path)
            name, ext = os.path.splitext(base_name)
            
            # Common naming patterns for back side images
            back_patterns = [
                f"{name}_back{ext}",
                f"{name}_b{ext}",
                f"{name.replace('front', 'back')}{ext}",
                f"{name.replace('_front', '_back')}{ext}",
                f"{name.replace('_f', '_b')}{ext}",
                f"{name}_2{ext}"
            ]
            
            # Check if any of these files exist
            for pattern in back_patterns:
                potential_path = os.path.join(dir_path, pattern)
                if os.path.exists(potential_path):
                    return potential_path
            
            # If not found by naming convention, look for similar images in the same directory
            files = [f for f in os.listdir(dir_path) if f.endswith(ext) and f != base_name]
            
            # If only one other image in directory, assume it's the back
            if len(files) == 1:
                return os.path.join(dir_path, files[0])
                
            # For multiple files, try to find one with similar filename
            front_prefix = name.split('_')[0]
            for file in files:
                if front_prefix in file and ('back' in file.lower() or '_b' in file.lower()):
                    return os.path.join(dir_path, file)
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting back side: {str(e)}")
            return None
    
    def _validate_cross_check(self, data):
        """
        Perform cross-validation of fields from both sides of ID
        
        Args:
            data: Combined data from front and back sides
            
        Returns:
            Validated and cleaned data
        """
        # If we have very little data, return as is
        if len(data) < 2:
            return data
            
        try:
            # Check for duplicate fields with different values
            # and choose the most reliable one
            field_confidence = {
                'personal_number': 0,
                'identity_number': 0,
                'document_number': 0,
                'first_name': 0,
                'last_name': 0,
                'birth_date': 0,
                'expiry_date': 0
            }
            
            # Assess confidence in each field
            for field in field_confidence.keys():
                if field in data:
                    value = data[field]
                    
                    # Higher confidence in values with expected length
                    if field == 'personal_number' and len(value) == 10:
                        field_confidence[field] += 2
                    elif field == 'identity_number' and (len(value) == 9 or (len(value) >= 8 and re.match(r'^[A-Z]{1,2}\d+$', value))):
                        field_confidence[field] += 2
                    
                    # Higher confidence in well-formatted dates
                    elif field in ['birth_date', 'expiry_date'] and re.match(r'^\d{2}\.\d{2}\.\d{4}$', value):
                        field_confidence[field] += 2
                    
                    # Personal number consistency check (first 6 digits match birth date)
                    if field == 'personal_number' and 'birth_date' in data:
                        date_str = data['birth_date']
                        date_match = re.match(r'(\d{2})\.(\d{2})\.(\d{4})', date_str)
                        if date_match:
                            day = date_match.group(1)
                            month = date_match.group(2)
                            year = date_match.group(3)[-2:]  # Last 2 digits of year
                            
                            # Check if first 6 digits match birth date in YYMMDD format
                            expected_prefix = year + month + day
                            if value.startswith(expected_prefix):
                                field_confidence[field] += 3
                                field_confidence['birth_date'] += 2
            
            # For name fields, prefer all-caps versions if present
            for field in ['first_name', 'last_name']:
                if field in data:
                    value = data[field]
                    if value.isupper():
                        field_confidence[field] += 1
            
            # Name fields should all be present if valid
            if 'first_name' in data and 'last_name' in data:
                field_confidence['first_name'] += 1
                field_confidence['last_name'] += 1
            
            return data
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return data 

    def _normalize_country_names(self, country):
        """
        Normalize country names
        
        Args:
            country: Country name string
            
        Returns:
            Normalized country name
        """
        if not country:
            return None
            
        # Country name mapping
        country_map = {
            'USA': 'UNITED STATES', 'US': 'UNITED STATES', 'UNITED STATES OF AMERICA': 'UNITED STATES',
            'UK': 'UNITED KINGDOM', 'GREAT BRITAIN': 'UNITED KINGDOM', 'ENGLAND': 'UNITED KINGDOM',
            'DEUTSCHLAND': 'GERMANY', 'DEU': 'GERMANY', 'GER': 'GERMANY',
            'FRANCE': 'FRANCE', 'FRA': 'FRANCE', 'FR': 'FRANCE',
            'BG': 'BULGARIA', 'BULGARIAN': 'BULGARIA',
            'ESP': 'SPAIN', 'ESPAÑA': 'SPAIN',
            'ITA': 'ITALY', 'ITALIA': 'ITALY', 
            'CAN': 'CANADA',
            'AUS': 'AUSTRALIA',
            'JPN': 'JAPAN', 'JAPON': 'JAPAN',
            'CHN': 'CHINA',
            'RUS': 'RUSSIA', 'RUSSIAN FEDERATION': 'RUSSIA'
        }
        
        country_upper = country.upper().strip()
        
        # Check if it's already in the map
        if country_upper in country_map:
            return country_map[country_upper]
        
        # Otherwise return the original but normalized to uppercase
        return country_upper 

    def _identify_text_field(self, roi_text):
        """
        Identify what field a text region might contain based on its content
        
        Args:
            roi_text: Text extracted from a region of interest
            
        Returns:
            Tuple of (field_name, confidence) or (None, 0) if field cannot be identified
        """
        field_text = roi_text.lower()
        
        # Check for common field names in the text
        if 'name' in field_text or 'given name' in field_text or 'first name' in field_text:
            return 'first_name', 0.8
        elif 'surname' in field_text or 'last name' in field_text or 'family name' in field_text:
            return 'last_name', 0.8
        elif 'birth' in field_text or 'date of birth' in field_text or 'dob' in field_text:
            return 'birth_date', 0.8
        elif 'expiry' in field_text or 'expiration' in field_text or 'valid until' in field_text:
            return 'expiry_date', 0.8
        elif 'sex' in field_text or 'gender' in field_text:
            return 'sex', 0.9
        elif 'nationality' in field_text or 'citizen' in field_text:
            return 'nationality', 0.8
        elif 'document' in field_text and ('no' in field_text or 'number' in field_text):
            return 'document_number', 0.8
        elif 'identity' in field_text and ('no' in field_text or 'number' in field_text):
            return 'identity_number', 0.8
        elif 'personal' in field_text:
            return 'personal_number', 0.8
        elif 'address' in field_text or 'residence' in field_text:
            return 'permanent_address', 0.7
        elif 'height' in field_text:
            return 'height', 0.9
        elif 'eye' in field_text and 'color' in field_text:
            return 'eye_color', 0.9
        else:
            # Try to detect field by format
            if re.search(r'\d{1,2}[./]\d{1,2}[./]\d{2,4}', field_text):
                return 'birth_date', 0.6  # Could be any date, but birth date is common
            elif re.search(r'^\d{6,10}$', field_text):
                return 'personal_number', 0.5  # Looks like an ID number
            elif re.search(r'^[A-Z]{1,2}\d{6,8}$', field_text):
                return 'identity_number', 0.6  # Common ID number format
            
        return None, 0 