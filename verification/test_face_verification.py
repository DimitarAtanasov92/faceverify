import os
import cv2
from face_verification import FaceVerification
import tempfile
from deepface import DeepFace

def test_face_verification():
    # Initialize the face verification class
    face_verifier = FaceVerification()
    
    selfie_path = r"C:\Users\EVLVS99\Desktop\face_veref\faceverify2\revolutlite2\media\verification_docs\user_29\c50d0a61-b288-49b8-8661-94e28a945025.jpg"
    id_card_path = r"C:\Users\EVLVS99\Desktop\face_veref\faceverify2\revolutlite2\media\verification_docs\user_29\50430101-d3e4-4ebd-bb77-7b698c6ee937.jpg"
    
    print("\n=== Testing Face Verification with Real Images ===")
    
    # Test 1: Face Detection on Selfie
    print("\nTest 1: Face Detection (Selfie)")
    if os.path.exists(selfie_path):
        try:
            faces = DeepFace.extract_faces(
                img_path=selfie_path,
                enforce_detection=True
            )
            print(f"✓ Face detection on selfie successful. Found {len(faces)} faces.")
        except Exception as e:
            print(f"✗ Face detection on selfie failed: {str(e)}")
    else:
        print(f"Selfie image not found: {selfie_path}")
    
    # Test 2: Face Detection on ID Card
    print("\nTest 2: Face Detection (ID Card)")
    if os.path.exists(id_card_path):
        try:
            faces = DeepFace.extract_faces(
                img_path=id_card_path,
                enforce_detection=True
            )
            print(f"✓ Face detection on ID card successful. Found {len(faces)} faces.")
        except Exception as e:
            print(f"✗ Face detection on ID card failed: {str(e)}")
    else:
        print(f"ID card image not found: {id_card_path}")
    
    # Test 3: Face Verification
    print("\nTest 3: Face Verification (Selfie vs ID Card)")
    if os.path.exists(selfie_path) and os.path.exists(id_card_path):
        try:
            is_match, confidence = face_verifier.verify_faces(selfie_path, id_card_path)
            print(f"✓ Face verification completed.")
            print(f"   Match: {is_match}")
            print(f"   Confidence: {confidence:.2f}")
        except Exception as e:
            print(f"✗ Face verification failed: {str(e)}")
    else:
        print("One or both images not found for verification.")
    
    # Test 4: Face Box Drawing
    print("\nTest 4: Face Box Drawing (Selfie)")
    with tempfile.TemporaryDirectory() as temp_dir:
        if os.path.exists(selfie_path):
            try:
                output_path = os.path.join(temp_dir, "selfie_face_boxes.jpg")
                face_verifier.draw_face_boxes(selfie_path, output_path)
                print(f"✓ Face box drawing on selfie completed. Output saved to: {output_path}")
            except Exception as e:
                print(f"✗ Face box drawing on selfie failed: {str(e)}")
        else:
            print(f"Selfie image not found: {selfie_path}")
        if os.path.exists(id_card_path):
            try:
                output_path = os.path.join(temp_dir, "idcard_face_boxes.jpg")
                face_verifier.draw_face_boxes(id_card_path, output_path)
                print(f"✓ Face box drawing on ID card completed. Output saved to: {output_path}")
            except Exception as e:
                print(f"✗ Face box drawing on ID card failed: {str(e)}")
        else:
            print(f"ID card image not found: {id_card_path}")

if __name__ == "__main__":
    test_face_verification() 