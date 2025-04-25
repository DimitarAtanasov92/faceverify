import cv2
import numpy as np
import os
import logging
import easyocr
import time
import re
from skimage import exposure
from PIL import Image
import pytesseract  # Keep as fallback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ai_ocr')

class AIOCR:
    """
    Advanced AI-powered OCR using EasyOCR with preprocessing techniques
    """
    def __init__(self, languages=['en'], gpu=False, debug_mode=False):
        self.debug_mode = debug_mode
        self.languages = languages
        self.gpu = gpu
        
        # Initialize EasyOCR reader
        logger.info(f"Initializing EasyOCR with languages: {languages}, GPU: {gpu}")
        try:
            self.reader = easyocr.Reader(languages, gpu=gpu)
            self.ocr_available = True
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            self.ocr_available = False
            logger.error(f"Failed to initialize EasyOCR: {str(e)}")
            
        # Initialize Tesseract as fallback
        self.tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if os.path.exists(self.tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
            self.tesseract_available = True
            logger.info(f"Tesseract OCR initialized as fallback at {self.tesseract_path}")
        else:
            self.tesseract_available = False
            logger.warning("Tesseract OCR not available as fallback")
            
        # Common patterns to extract data
        self.field_patterns = {
            'date': r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'id_number': r'\b[A-Z0-9]{6,12}\b',
            'amount': r'\b\$?\s?\d+(?:,\d{3})*(?:\.\d{2})?\b'
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
            
    def extract_text(self, image_path, preprocess_level='default', confidence_threshold=0.4):
        """
        Extract text from an image using EasyOCR with preprocessing
        
        Args:
            image_path: Path to the image file
            preprocess_level: Level of preprocessing ('minimal', 'default', 'aggressive', 'adaptive')
            confidence_threshold: Minimum confidence level for results
            
        Returns:
            Dictionary with extracted text and confidence scores
        """
        start_time = time.time()
        
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(image_path, preprocess_level)
            if processed_img is None:
                logger.error("Preprocessing failed")
                return {"text": "", "confidence": 0, "error": "Preprocessing failed"}
                
            # Check if EasyOCR is available
            if not self.ocr_available:
                if self.tesseract_available:
                    logger.warning("Using Tesseract OCR as fallback")
                    text = pytesseract.image_to_string(processed_img, lang=self.languages[0])
                    return {"text": text, "confidence": 0.5, "method": "tesseract"}
                else:
                    return {"text": "", "confidence": 0, "error": "No OCR engine available"}
                    
            # Use EasyOCR for text extraction
            results = self.reader.readtext(processed_img)
            
            # Filter by confidence threshold and combine results
            filtered_results = [item for item in results if item[2] >= confidence_threshold]
            
            # Extract full text and calculate average confidence
            full_text = " ".join([item[1] for item in filtered_results])
            avg_confidence = sum([item[2] for item in filtered_results]) / len(filtered_results) if filtered_results else 0
            
            # Extract detailed results
            detailed_results = [
                {"text": item[1], "confidence": item[2], "bbox": item[0]} 
                for item in results
            ]
            
            # Sort detailed results by Y-coordinate for natural reading order
            detailed_results.sort(key=lambda x: x["bbox"][0][1])  # Sort by Y of top-left corner
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "detailed_results": detailed_results,
                "process_time": process_time,
                "method": "easyocr"
            }
            
        except Exception as e:
            logger.error(f"Error in extract_text: {str(e)}")
            
            # Try fallback to Tesseract if available
            if self.tesseract_available:
                try:
                    logger.warning("Using Tesseract OCR as fallback after EasyOCR error")
                    pil_img = Image.fromarray(processed_img) if processed_img is not None else Image.open(image_path)
                    text = pytesseract.image_to_string(pil_img, lang=self.languages[0])
                    return {"text": text, "confidence": 0.5, "method": "tesseract", "error": str(e)}
                except Exception as te:
                    logger.error(f"Tesseract fallback also failed: {str(te)}")
                    
            return {"text": "", "confidence": 0, "error": str(e)}
            
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
                fields = self.extract_fields(ocr_result["text"], custom_patterns)
                ocr_result["extracted_fields"] = fields
                
            return ocr_result
            
        except Exception as e:
            logger.error(f"Error in read_document: {str(e)}")
            return {"error": str(e)}
            
    def extract_tables(self, image_path, preprocess_level='default'):
        """
        Extract tabular data from images
        
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
                
            # Get text with bounding boxes from EasyOCR
            if not self.ocr_available:
                return {"error": "EasyOCR not available"}
                
            # Extract text with positions
            results = self.reader.readtext(processed_img)
            
            # Group text blocks by rows based on y-coordinates
            tolerance = 20  # Pixels tolerance for same row
            rows = []
            current_row = []
            last_y = -tolerance * 2
            
            # Sort by Y coordinate first
            sorted_results = sorted(results, key=lambda x: (x[0][0][1] + x[0][2][1]) / 2)  # Average Y of top-left and bottom-left
            
            for result in sorted_results:
                bbox = result[0]
                text = result[1]
                confidence = result[2]
                
                # Calculate center Y coordinate
                center_y = (bbox[0][1] + bbox[2][1]) / 2
                
                # If this block is on a new row
                if abs(center_y - last_y) > tolerance:
                    if current_row:
                        # Sort the current row by X coordinate
                        current_row.sort(key=lambda x: x["bbox"][0][0])  # Sort by X of top-left corner
                        rows.append(current_row)
                        current_row = []
                    last_y = center_y
                    
                # Add to current row
                current_row.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": bbox
                })
                
            # Add the last row
            if current_row:
                current_row.sort(key=lambda x: x["bbox"][0][0])  # Sort by X
                rows.append(current_row)
                
            # Now rows contains the table-like data
            table_data = [
                [cell["text"] for cell in row]
                for row in rows
            ]
            
            return {
                "table_data": table_data,
                "row_count": len(table_data),
                "detected_rows": rows
            }
            
        except Exception as e:
            logger.error(f"Error extracting tables: {str(e)}")
            return {"error": str(e)}
    
    def recognize_handwriting(self, image_path, preprocess_level='adaptive'):
        """
        Optimized method for handwriting recognition
        
        Args:
            image_path: Path to the image file
            preprocess_level: Level of preprocessing
            
        Returns:
            Extracted text from handwriting
        """
        try:
            # Use adaptive preprocessing which works better for handwriting
            processed_img = self.preprocess_image(image_path, preprocess_level)
            
            # Use EasyOCR with handwriting optimization
            if not self.ocr_available:
                return {"error": "EasyOCR not available"}
            
            # Lower confidence threshold for handwriting
            results = self.reader.readtext(processed_img, paragraph=True)
            
            full_text = " ".join([item[1] for item in results])
            avg_confidence = sum([item[2] for item in results]) / len(results) if results else 0
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "method": "easyocr_handwriting"
            }
            
        except Exception as e:
            logger.error(f"Error recognizing handwriting: {str(e)}")
            return {"error": str(e)} 