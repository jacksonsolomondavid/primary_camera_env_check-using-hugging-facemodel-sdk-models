import cv2
import numpy as np
from PIL import Image # Required for Hugging Face models to load images

# Import Hugging Face components
from transformers import pipeline

# Load Hugging Face Zero-Shot Object Detection pipeline once globally.
# This model will download the first time it's run.
hf_detector_earphones = pipeline(model="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection")


# --- Earphone Detection Function (Hugging Face Model) ---
def detect_earphones_insight(image_path, darkness_threshold=0.15, zoom_factor=1.4, apply_clahe=True, apply_sharpening=False):
    """
    Detects the presence of earphones/headphones in an image using a Hugging Face zero-shot object detection model.
    Returns a dictionary with "earphones_detected": True/False.

    Args:
        image_path (str): Path to the input image.
        darkness_threshold (float): (Not directly used by Hugging Face pipeline).
        zoom_factor (float): (Not directly used by Hugging Face pipeline).
        apply_clahe (bool): (Not directly used by Hugging Face pipeline).
        apply_sharpening (bool): (Not directly used by Hugging Face pipeline).

    Returns:
        dict: A dictionary containing the earphone detection status:
              - "earphones_detected": True if earphones/headphones are detected, False otherwise.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not load image from {image_path}")
        return {"earphones_detected": False}
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return {"earphones_detected": False}

    # Define text queries for earphones/headphones.
    # Removed "not a hand gesture" as it can confuse the model.
    # Added more specific terms.
    text_queries = ["earphone", "headphone", "earbud", "wireless earphone", "headset", "earpiece", "in-ear headphone"]
    
    # Set a higher detection confidence threshold to reduce false positives.
    # This is a crucial parameter to tune. Start higher and reduce if needed.
    detection_threshold = 0.3 # Increased threshold to reduce false positives

    # Perform zero-shot object detection
    predictions = hf_detector_earphones(image, candidate_labels=text_queries)

    earphone_detected = False
    
    # Optional: Print all detections for debugging purposes
    print(f"\n--- Raw Hugging Face Detections for {image_path} ---")
    for p in predictions:
        print(f"Detected: {p['label']}, Score: {p['score']:.2f}, Box: {p['box']}")

    for p in predictions:
        if p['score'] > detection_threshold:
            # If any object is detected matching the query above the threshold
            earphone_detected = True
            break # Found an earphone-like object, no need to check further

    return {
        "earphones_detected": earphone_detected
    }