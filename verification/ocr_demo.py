import os
import argparse
import json
import time
import cv2
from ai_ocr import AIOCR
from django.conf import settings
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add the Django project root to sys.path
# Adjust the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "revolutlite2.settings")

def visualize_results(image_path, ocr_results, output_path=None):
    """
    Visualize OCR results on the original image
    """
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img_rgb)
    
    # Random colors for bounding boxes
    colors = plt.cm.rainbow(np.linspace(0, 1, 20))
    
    # Draw detected text regions
    if 'detailed_results' in ocr_results:
        for i, det in enumerate(ocr_results['detailed_results']):
            bbox = det['bbox']
            text = det['text']
            conf = det['confidence']
            
            # Convert points to numpy array
            bbox_np = np.array(bbox).astype(np.int32)
            
            # Get random color
            color = tuple(colors[i % len(colors)][:3])
            
            # Draw box
            cv2.polylines(img_rgb, [bbox_np], True, color, 2)
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            pos = bbox_np[0]
            cv2.putText(img_rgb, f"{text} ({conf:.2f})", 
                      (pos[0], pos[1] - 10), font, 0.5, color, 2)
    
    # Show image
    ax.imshow(img_rgb)
    ax.set_title(f"OCR Results - {len(ocr_results.get('detailed_results', []))} text regions found")
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="AI OCR Demo Script")
    parser.add_argument('--image', '-i', required=True, help='Path to the image file')
    parser.add_argument('--preprocess', '-p', default='default', 
                      choices=['minimal', 'default', 'aggressive', 'adaptive'],
                      help='Preprocessing level')
    parser.add_argument('--mode', '-m', default='general', 
                      choices=['general', 'handwriting', 'table'],
                      help='OCR mode')
    parser.add_argument('--output', '-o', default=None, help='Output path for JSON results')
    parser.add_argument('--visualize', '-v', action='store_true', help='Visualize results')
    parser.add_argument('--gpu', '-g', action='store_true', help='Use GPU if available')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Check if the image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1
        
    print(f"Initializing AI OCR model...")
    ocr = AIOCR(languages=['en'], gpu=args.gpu, debug_mode=args.debug)
    
    print(f"Processing image: {args.image}")
    start_time = time.time()
    
    # Process based on selected mode
    if args.mode == 'general':
        results = ocr.read_document(args.image, preprocess_level=args.preprocess)
    elif args.mode == 'handwriting':
        results = ocr.recognize_handwriting(args.image, preprocess_level=args.preprocess)
    elif args.mode == 'table':
        results = ocr.extract_tables(args.image, preprocess_level=args.preprocess)
    
    processing_time = time.time() - start_time
    
    # Add processing time and arguments to results
    results['processing_time'] = processing_time
    results['args'] = vars(args)
    
    # Print summary
    print(f"Processing completed in {processing_time:.2f} seconds")
    if 'text' in results:
        print(f"Extracted text ({len(results['text'])} chars):")
        print(f"{results['text'][:1000]}...")
        if len(results['text']) > 1000:
            print(f"... (truncated, {len(results['text'])} total characters)")
    
    if 'extracted_fields' in results:
        print("\nExtracted Fields:")
        for field, value in results['extracted_fields'].items():
            print(f"  - {field}: {value}")
    
    if 'table_data' in results:
        print(f"\nDetected table with {results['row_count']} rows:")
        for i, row in enumerate(results['table_data'][:5]):
            print(f"  Row {i+1}: {row}")
        if results['row_count'] > 5:
            print(f"  ... (and {results['row_count'] - 5} more rows)")
    
    # Save results to file if output path provided
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {args.output}")
    
    # Visualize results if requested
    if args.visualize:
        vis_output = None
        if args.output:
            # Create visualization with same basename as output but with .png extension
            vis_output = os.path.splitext(args.output)[0] + "_viz.png"
        
        visualize_results(args.image, results, vis_output)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 