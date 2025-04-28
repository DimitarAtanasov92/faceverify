from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
import os
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

class FaceVerification:
    def __init__(
        self,
        threshold: float = 0.6,
        model_name: str = "VGG-Face",
        enforce_detection: bool = False,
        detector_backend: str = "opencv",
        distance_metric: str = "cosine"
    ):
        """
        Initialize the FaceVerification class with configurable parameters.
        
        Args:
            threshold (float): Threshold for verification (0.0 to 1.0)
            model_name (str): Face recognition model to use
            enforce_detection (bool): Whether to enforce face detection
            detector_backend (str): Face detector backend to use
            distance_metric (str): Distance metric for face comparison
        """
        self.threshold = threshold
        self.model_name = model_name
        self.enforce_detection = enforce_detection
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image using cv2 with validation.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[np.ndarray]: Loaded image array or None if loading fails
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            return None
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
            return image
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    def verify_faces(self, selfie_path: str, id_document_path: str) -> Tuple[bool, float]:
        """
        Compare faces from selfie and ID document.
        
        Args:
            selfie_path (str): Path to the selfie image
            id_document_path (str): Path to the ID document image
            
        Returns:
            Tuple[bool, float]: (is_match, confidence)
                - is_match: True if faces match, False otherwise
                - confidence: Similarity score between 0 and 1
        """
        if not all(os.path.exists(path) for path in [selfie_path, id_document_path]):
            self.logger.error("One or both image paths do not exist")
            return False, 0.0

        try:
            # Use DeepFace to verify
            result = DeepFace.verify(
                img1_path=selfie_path,
                img2_path=id_document_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=self.enforce_detection
            )
            
            is_match = result["verified"]
            confidence = result.get("distance", 0.0)
            
            # Convert distance to similarity score (closer to 1 means more similar)
            confidence = 1 - min(confidence, 1.0)
            
            self.logger.info(f"Face verification completed. Match: {is_match}, Confidence: {confidence:.2f}")
            return is_match, confidence

        except Exception as e:
            self.logger.error(f"Error during face verification: {str(e)}")
            return False, 0.0

    def draw_face_boxes(
        self, 
        image_path: str, 
        output_path: str,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        box_thickness: int = 2
    ) -> bool:
        """
        Draw boxes around detected faces and save the image.
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save annotated image
            box_color (Tuple[int, int, int]): RGB color for boxes (default: green)
            box_thickness (int): Thickness of box lines
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(image_path):
            self.logger.error(f"Input image not found: {image_path}")
            return False

        try:
            # Use DeepFace to detect faces
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False

            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=self.enforce_detection
            )
            
            # Draw boxes around detected faces
            for face in faces:
                facial_area = face["facial_area"]
                x = facial_area["x"]
                y = facial_area["y"]
                w = facial_area["w"]
                h = facial_area["h"]
                
                cv2.rectangle(image, (x, y), (x + w, y + h), box_color, box_thickness)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            cv2.imwrite(output_path, image)
            self.logger.info(f"Successfully saved annotated image to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error drawing face boxes: {str(e)}")
            # If face detection fails, try to save the original image
            try:
                cv2.imwrite(output_path, cv2.imread(image_path))
                self.logger.warning("Saved original image due to face detection failure")
                return True
            except Exception as e:
                self.logger.error(f"Failed to save original image: {str(e)}")
                return False 