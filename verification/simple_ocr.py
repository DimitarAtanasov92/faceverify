import cv2
import numpy as np
import os
import logging
import re
import time
from skimage import exposure
from PIL import Image
import pytesseract

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_ocr')

class SimpleOCR:
    """
    OCR implementation using Tesseract and OpenCV
    """
    def __init__(self, languages=['eng'], debug_mode=False):
        self.debug_mode = debug_mode
        self.languages = languages
        
        # Initialize Tesseract path
        self.tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(self.tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            self.tesseract_available = True
            logger.info(f"Tesseract OCR initialized at {self.tesseract_path}")
        else:
            self.tesseract_available = False
            logger.warning("Tesseract OCR not available")
            
        # Common patterns to extract data
        self.field_patterns = {
            'date': r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'id_number': r'\b[A-Z0-9]{6,12}\b',
            'amount': r'\b\$?\s?\d+(?:,\d{3})*(?:\.\d{2})?\b'
        }
        
        # Document-specific patterns
        self.document_patterns = {
            'first_name': r'(?:Given|First|Christian)\s*names?\s*[:/]?\s*([A-Z]+(?:\s[A-Z]+)?)',
            'last_name': r'(?:Surname|Last\s*name|Family\s*name)\s*[:/]?\s*([A-Z]+(?:\s[A-Z]+)?)',
            'personal_number': r'(?:Personal [Nn]umber|PIN)\s*[:/]?\s*(\d+)',
            'identity_number': r'(?:Identity [Nn]umber|ID|№|Number)\s*[:/]?\s*([A-Z0-9]+)',
            'document_number': r'(?:Document [Nn]o\.?|№)\s*[:/]?\s*([A-Z0-9]+)',
            'birth_date': r'(?:Date\s*of\s*birth|Birth\s*date)\s*[:/]?\s*(\d{1,2}[.\/-]\d{1,2}[.\/-]\d{2,4})',
            'expiry_date': r'(?:Date\s*of\s*expiry|Expiry\s*date|Valid\s*until)\s*[:/]?\s*(\d{1,2}[.\/-]\d{1,2}[.\/-]\d{2,4})',
            'nationality': r'(?:Nationality|Citizen\s*of)\s*[:/]?\s*([A-Z]+)',
        }
            
    def preprocess_image(self, image_path, preprocessing_level='default'):
        """
        Preprocess the image for better OCR results
        
        Args:
            image_path: Path to the image file
            preprocessing_level: Level of preprocessing ('minimal', 'default', 'aggressive', 'adaptive')
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to read image from {image_path}")
                return None
                
            # Store original for comparison if in debug mode
            if self.debug_mode:
                original = img.copy()
                
            # Auto-correct rotation if needed
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
                # Enhance contrast using CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                
                # Apply bilateral filter to reduce noise while preserving edges
                filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
                
                if self.debug_mode:
                    self._save_debug_image(filtered, image_path, "adaptive")
                
                return filtered
                
            else:
                logger.warning(f"Unknown preprocessing level: {preprocessing_level}, using default")
                return gray
                
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            # Return original image on error
            return cv2.imread(image_path)
            
    def _correct_rotation(self, image):
        """
        Auto-detect and correct image rotation
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Use Hough Line Transform to detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
            
            if lines is None or len(lines) == 0:
                return image  # No lines detected, return original
                
            # Calculate the angle of each line
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:  # Avoid division by zero
                    continue
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                angles.append(angle)
                
            if not angles:
                return image  # No valid angles, return original
                
            # Get the median angle
            median_angle = np.median(angles)
            
            # If angle is close to vertical, adjust it
            if abs(median_angle) > 45:
                median_angle = 90 - abs(median_angle)
                if angles[0] < 0:
                    median_angle = -median_angle
                    
            # Skip rotation for small angles
            if abs(median_angle) < 1:
                return image
                
            # Rotate the image
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
        except Exception as e:
            logger.error(f"Error correcting rotation: {str(e)}")
            return image  # Return original image on error
            
    def _save_debug_image(self, img, original_path, suffix):
        """
        Save intermediate images for debugging
        """
        if not self.debug_mode:
            return
            
        try:
            filename, ext = os.path.splitext(original_path)
            debug_path = f"{filename}_{suffix}_debug{ext}"
            cv2.imwrite(debug_path, img)
            logger.debug(f"Saved debug image: {debug_path}")
        except Exception as e:
            logger.error(f"Error saving debug image: {str(e)}")
            
    def extract_text(self, image_path, preprocess_level='default'):
        """
        Extract text from an image using Tesseract OCR with preprocessing
        
        Args:
            image_path: Path to the image file
            preprocess_level: Level of preprocessing ('minimal', 'default', 'aggressive', 'adaptive')
            
        Returns:
            Dictionary with extracted text and confidence scores
        """
        start_time = time.time()
        
        try:
            # Check if Tesseract is available
            if not self.tesseract_available:
                return {"text": "", "error": "Tesseract OCR not available"}
                
            # Preprocess the image
            processed_img = self.preprocess_image(image_path, preprocess_level)
            if processed_img is None:
                logger.error("Preprocessing failed")
                return {"text": "", "error": "Preprocessing failed"}
            
            # Convert to PIL Image as required by pytesseract
            pil_img = Image.fromarray(processed_img)
            
            # Extract text with Tesseract
            text = pytesseract.image_to_string(pil_img, lang=self.languages[0])
            
            # Get confidence scores for each word
            data = pytesseract.image_to_data(pil_img, lang=self.languages[0], output_type=pytesseract.Output.DICT)
            
            # Calculate average confidence
            confidences = [float(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Create detailed results with bounding boxes
            detailed_results = []
            for i in range(len(data['text'])):
                if data['text'][i].strip() and int(data['conf'][i]) > 0:
                    detailed_results.append({
                        "text": data['text'][i],
                        "confidence": float(data['conf'][i]) / 100,  # Convert to 0-1 scale
                        "bbox": [
                            [data['left'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                            [data['left'][i], data['top'][i] + data['height'][i]]
                        ]
                    })
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            return {
                "text": text,
                "confidence": avg_confidence / 100,  # Convert to 0-1 scale
                "detailed_results": detailed_results,
                "process_time": process_time,
                "method": "tesseract"
            }
            
        except Exception as e:
            logger.error(f"Error in extract_text: {str(e)}")
            return {"text": "", "error": str(e)}
            
    def extract_fields(self, text, field_patterns=None):
        """
        Extract structured data from extracted text using regex patterns
        
        Args:
            text: Extracted text
            field_patterns: Custom patterns to use (optional)
            
        Returns:
            Dictionary of extracted fields
        """
        if not text:
            return {}
            
        # Use provided patterns or default ones
        patterns = field_patterns or self.field_patterns
        
        extracted_fields = {}
        
        # Apply each pattern to extract data
        for field_name, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                extracted_fields[field_name] = matches[0]  # Take first match
                
        # Try document-specific patterns
        for field_name, pattern in self.document_patterns.items():
            matches = re.findall(pattern, text)
            if matches and field_name not in extracted_fields:
                extracted_fields[field_name] = matches[0]
                
        return extracted_fields
        
    def read_document(self, image_path, preprocess_level='default', custom_patterns=None):
        """
        Full document reading workflow: preprocess, OCR, extract fields
        
        Args:
            image_path: Path to the image file
            preprocess_level: Level of preprocessing
            custom_patterns: Custom regex patterns for field extraction
            
        Returns:
            Dictionary with extracted text and fields
        """
        try:
            # Extract text from image
            ocr_result = self.extract_text(image_path, preprocess_level)
            
            # Extract structured fields from text
            if ocr_result.get("text"):
                patterns = custom_patterns or {**self.field_patterns, **self.document_patterns}
                fields = self.extract_fields(ocr_result["text"], patterns)
                ocr_result["extracted_fields"] = fields
                
            return ocr_result
            
        except Exception as e:
            logger.error(f"Error in read_document: {str(e)}")
            return {"error": str(e)}
            
    def extract_tables(self, image_path, preprocess_level='default'):
        """
        Extract tabular data from images using Tesseract
        
        Args:
            image_path: Path to the image file
            preprocess_level: Level of preprocessing
            
        Returns:
            List of detected tables with cell content
        """
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(image_path, preprocess_level)
            if processed_img is None:
                return {"error": "Preprocessing failed"}
                
            # Check if Tesseract is available
            if not self.tesseract_available:
                return {"error": "Tesseract OCR not available"}
            
            # Convert to PIL Image as required by pytesseract
            pil_img = Image.fromarray(processed_img)
            
            # Get data with bounding boxes
            data = pytesseract.image_to_data(pil_img, lang=self.languages[0], output_type=pytesseract.Output.DICT)
            
            # Group text elements by line (approximated by top position)
            lines = {}
            for i in range(len(data['text'])):
                if data['text'][i].strip() and int(data['conf'][i]) > 0:
                    top = data['top'][i]
                    # Group by top position within a tolerance
                    line_key = round(top / 10) * 10  # Approximate to nearest 10px
                    if line_key not in lines:
                        lines[line_key] = []
                    
                    lines[line_key].append({
                        "text": data['text'][i],
                        "confidence": float(data['conf'][i]) / 100,
                        "bbox": [
                            [data['left'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i]],
                            [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                            [data['left'][i], data['top'][i] + data['height'][i]]
                        ],
                        "left": data['left'][i]
                    })
            
            # Sort lines by top position
            sorted_line_keys = sorted(lines.keys())
            
            # For each line, sort elements by left position
            rows = []
            for key in sorted_line_keys:
                # Sort elements in this line by left position
                sorted_line = sorted(lines[key], key=lambda x: x["left"])
                rows.append([element["text"] for element in sorted_line])
            
            return {
                "table_data": rows,
                "row_count": len(rows),
                "method": "tesseract"
            }
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return {"error": str(e)}
    
    def recognize_handwriting(self, image_path, preprocess_level='adaptive'):
        """
        Attempt to recognize handwriting using optimized preprocessing
        
        Args:
            image_path: Path to the image file
            preprocess_level: Level of preprocessing
            
        Returns:
            Extracted text from handwriting
        """
        try:
            # Use adaptive preprocessing which works better for handwriting
            processed_img = self.preprocess_image(image_path, preprocess_level)
            
            # Check if Tesseract is available
            if not self.tesseract_available:
                return {"error": "Tesseract OCR not available"}
            
            # Convert to PIL Image as required by pytesseract
            pil_img = Image.fromarray(processed_img)
            
            # Use Tesseract with optimized config for handwriting
            config = '--psm 6'  # Assume a single block of text
            text = pytesseract.image_to_string(pil_img, lang=self.languages[0], config=config)
            
            return {
                "text": text,
                "confidence": 0.5,  # Default confidence for handwriting - Tesseract is not ideal for this
                "method": "tesseract_handwriting"
            }
            
        except Exception as e:
            logger.error(f"Error recognizing handwriting: {str(e)}")
            return {"error": str(e)}
            
    def process_id_card(self, front_image_path, back_image_path=None, debug_level='normal'):
        """
        Process ID card images and extract key information
        
        Args:
            front_image_path: Path to front image
            back_image_path: Path to back image (optional)
            debug_level: Level of debugging info
            
        Returns:
            Dictionary of extracted fields
        """
        try:
            # Process front image
            preprocess_level = 'adaptive' if debug_level == 'verbose' else 'default'
            front_result = self.read_document(front_image_path, preprocess_level)
            
            # Extract fields from front
            extracted_data = front_result.get('extracted_fields', {})
            
            # Process back image if available
            if back_image_path:
                back_result = self.read_document(back_image_path, preprocess_level)
                # Merge with front image results
                back_fields = back_result.get('extracted_fields', {})
                for field, value in back_fields.items():
                    if field not in extracted_data or not extracted_data[field]:
                        extracted_data[field] = value
            
            # Try handwriting recognition as a fallback if few fields were found
            if len(extracted_data) < 2 and debug_level in ['normal', 'verbose']:
                handwriting_result = self.recognize_handwriting(front_image_path)
                if 'text' in handwriting_result and handwriting_result['text'].strip():
                    handwriting_fields = self.extract_fields(handwriting_result['text'])
                    # Add any new fields found
                    for field, value in handwriting_fields.items():
                        if field not in extracted_data or not extracted_data[field]:
                            extracted_data[field] = value
            
            # Add raw text for debugging
            if debug_level in ['normal', 'verbose']:
                if 'text' in front_result:
                    extracted_data['raw_front_text'] = front_result['text']
                if back_image_path and 'text' in back_result:
                    extracted_data['raw_back_text'] = back_result['text']
                
                # Store processing info for debugging
                extracted_data['processing_time'] = front_result.get('process_time', 0)
                extracted_data['confidence'] = front_result.get('confidence', 0)
                
            return extracted_data
            
        except Exception as e:
            logger.error(f"Error in process_id_card: {str(e)}")
            return {"error": str(e)} 