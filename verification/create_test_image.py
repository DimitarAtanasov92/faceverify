import cv2
import numpy as np

def create_test_image():
    # Create a blank image
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    image.fill(255)  # White background
    
    # Draw a simple face
    # Face
    cv2.circle(image, (250, 250), 100, (0, 0, 0), 2)
    
    # Eyes
    cv2.circle(image, (200, 220), 15, (0, 0, 0), -1)
    cv2.circle(image, (300, 220), 15, (0, 0, 0), -1)
    
    # Mouth
    cv2.ellipse(image, (250, 280), (50, 20), 0, 0, 180, (0, 0, 0), 2)
    
    # Save the image
    cv2.imwrite("test_face.jpg", image)
    print("Test image created: test_face.jpg")

if __name__ == "__main__":
    create_test_image() 