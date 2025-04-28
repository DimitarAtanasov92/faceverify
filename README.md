# Face Verification System

A Django-based face verification system that combines OCR and facial recognition capabilities for identity verification.

## Features

- Face detection and verification using DeepFace
- OCR (Optical Character Recognition) for document processing
- User authentication and management
- Web-based interface for easy interaction
- Secure storage of user data and media

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine installed on your system
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DimitarAtanasov92/faceverify.git
cd faceverify2
```

2. Install dependencies:
```bash
cd revolutlite2
pip install -r requirements.txt
```

3. Set up the database:
```bash
python manage.py migrate
```

4. Create a superuser (optional):
```bash
python manage.py createsuperuser
```

## Running the Application

1. Start the development server:
```bash
python manage.py runserver
```

2. Access the application at `http://127.0.0.1:8000/`

## Project Structure

- `revolutlite2/` - Main project directory
  - `verification/` - Face verification app
  - `ocr/` - OCR processing app
  - `users/` - User management app
  - `templates/` - HTML templates
  - `static/` - Static files (CSS, JS, images)
  - `media/` - User-uploaded media files

## Dependencies

- Django >= 5.0.0
- Pillow >= 9.5.0
- OpenCV >= 4.5.0
- Pytesseract >= 0.3.0
- NumPy >= 1.22.0
- DeepFace >= 0.0.75

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
