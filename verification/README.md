# Advanced OCR for Document Verification

This module provides advanced Optical Character Recognition (OCR) capabilities for the document verification system.

## Features

- **Advanced OCR**: Utilizes Tesseract OCR with enhanced image processing
- **Multiple Processing Modes**:
  - General document processing
  - Handwriting recognition
  - Table extraction
- **Advanced Image Preprocessing**:
  - Auto-rotation correction
  - Multiple preprocessing levels (minimal, default, aggressive, adaptive)
  - Noise reduction
  - Contrast enhancement
- **Field Extraction**: Automatic extraction of common fields from documents

## Installation

The OCR module requires Tesseract to be installed on your system:

- **Windows**: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

The default path is set to `C:\Program Files\Tesseract-OCR\tesseract.exe` for Windows. You may need to adjust this in the `simple_ocr.py` file if your installation is different.

## Usage

### In Django Views

```python
from verification.simple_ocr import SimpleOCR

# Initialize the OCR engine
ocr = SimpleOCR(languages=['eng'], debug_mode=True)

# Process a document image
result = ocr.read_document('path/to/image.jpg', preprocess_level='default')

# Extract text from the result
text = result.get('text', '')
print(f"Extracted Text: {text}")

# Extract fields from the result
fields = result.get('extracted_fields', {})
for field, value in fields.items():
    print(f"{field}: {value}")
```

### Standalone Testing

You can use the included test script to evaluate OCR performance:

```
python verification/test_simple_ocr.py path/to/image.jpg --mode general
```

## OCR Modes

- **General Mode**: Standard document processing
- **Handwriting Mode**: Optimized for handwritten text (limited accuracy)
- **Table Mode**: Extracts tabular data with row/column structure

## Preprocessing Levels

- **Minimal**: Only grayscale conversion
- **Default**: Thresholding and noise removal
- **Aggressive**: Adaptive thresholding and morphological operations
- **Adaptive**: CLAHE contrast enhancement and bilateral filtering

## Extracted Fields

The OCR engine can automatically extract the following fields:

- Dates in common formats
- Email addresses
- Phone numbers
- ID numbers (6-12 digit alphanumeric sequences)
- Currency amounts
- Document-specific fields (first name, last name, etc.)

You can add custom patterns in the `field_patterns` dictionary in the `SimpleOCR` class.

## Debug Mode

When debug mode is enabled:
- Intermediate processed images are saved next to the original
- More detailed logging is enabled
- Original text extraction results are preserved

## Performance Considerations

- Preprocessing level affects both accuracy and processing time
- For production use, consider setting up a background task for OCR processing

## Integration with Document Verification

The OCR module is integrated with the document verification workflow:

1. User uploads ID documents
2. OCR processes the images
3. Extracted data is stored and validated
4. User completes verification with selfie upload
5. System performs matching verification

---

For more information, see the code documentation in `simple_ocr.py`. 