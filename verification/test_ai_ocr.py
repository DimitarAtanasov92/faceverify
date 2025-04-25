#!/usr/bin/env python
"""
Standalone test script for AI OCR model.
Usage: python test_ai_ocr.py <path_to_image> [--mode general|handwriting|table]
"""

import os
import sys
import json
import argparse
from ai_ocr import AIOCR

def main():
    parser = argparse.ArgumentParser(description="Test AI OCR Model")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument("--mode", choices=["general", "handwriting", "table"], 
                       default="general", help="OCR mode to use")
    parser.add_argument("--preprocess", choices=["minimal", "default", "aggressive", "adaptive"],
                       default="default", help="Preprocessing level")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return 1
    
    # Initialize OCR with specified parameters
    print(f"Initializing AI OCR with {'GPU' if args.gpu else 'CPU'} mode...")
    ocr = AIOCR(languages=['en'], gpu=args.gpu, debug_mode=args.debug)
    
    # Process image based on the selected mode
    print(f"Processing image: {args.image_path}")
    print(f"Mode: {args.mode}, Preprocessing: {args.preprocess}")
    
    try:
        if args.mode == "general":
            result = ocr.read_document(args.image_path, preprocess_level=args.preprocess)
        elif args.mode == "handwriting":
            result = ocr.recognize_handwriting(args.image_path, preprocess_level=args.preprocess)
        elif args.mode == "table":
            result = ocr.extract_tables(args.image_path, preprocess_level=args.preprocess)
        else:
            print("Invalid mode")
            return 1
        
        # Print results
        print("\n--- OCR Results ---")
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return 1
        
        if "text" in result:
            print(f"\nExtracted Text ({len(result['text'])} characters):")
            print("-" * 50)
            print(result["text"])
            print("-" * 50)
        
        if "confidence" in result:
            print(f"\nConfidence: {result['confidence']:.2f}")
        
        if "method" in result:
            print(f"Method: {result['method']}")
        
        if "process_time" in result:
            print(f"Processing Time: {result['process_time']:.2f} seconds")
        
        if "extracted_fields" in result:
            print("\nExtracted Fields:")
            print("-" * 50)
            for field, value in result["extracted_fields"].items():
                print(f"{field}: {value}")
            print("-" * 50)
        
        if "table_data" in result:
            print("\nExtracted Table:")
            print("-" * 50)
            for row in result["table_data"]:
                print(" | ".join(row))
            print("-" * 50)
        
        # Save results to JSON if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.output}")
        
        return 0
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 