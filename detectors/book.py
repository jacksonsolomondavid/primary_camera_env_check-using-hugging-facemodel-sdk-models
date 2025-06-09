# device_detector.py

import cv2
import numpy as np
from PIL import Image # Required for Hugging Face models to load images
# from ultralytics import YOLO # YOLO is no longer used in this version for book detection

# Import Hugging Face components
from transformers import pipeline

# Load Hugging Face Zero-Shot Object Detection pipeline once globally.
# This model will download the first time it's run.
# 'google/owlv2-base-patch16-ensemble' is a good choice for zero-shot detection.
# You can explore others on Hugging Face Hub if needed.
hf_detector = pipeline(model="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection")

# --- Preprocessing Helper Function (No longer directly used by detect_book_huggingface) ---
# Keeping it here for other potential uses or if detect_devices is still in the file.
def preprocess_image(image, zoom_factor=1.0, apply_clahe=False, apply_sharpening=False):
    """
    Applies various preprocessing steps to an image.
    (Note: This function is not directly used by the Hugging Face book detection,
    as HF models handle their own internal preprocessing.)

    Args:
        image (np.array): The input image (OpenCV format).
        zoom_factor (float): Factor to resize the image (e.g., 1.5 for 50% larger).
        apply_clahe (bool): Whether to apply Contrast Limited Adaptive Histogram Equalization.
        apply_sharpening (bool): Whether to apply a simple sharpening filter.

    Returns:
        np.array: The preprocessed image.
    """
    processed_image = image.copy()

    if zoom_factor != 1.0:
        new_width = int(processed_image.shape[1] * zoom_factor)
        new_height = int(processed_image.shape[0] * zoom_factor)
        processed_image = cv2.resize(processed_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    if apply_clahe:
        lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(l_channel)
        lab[:, :, 0] = cl1
        processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if apply_sharpening:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        processed_image = cv2.filter2D(processed_image, -1, kernel)

    return processed_image

# --- Book Detection Function (Hugging Face Model) ---
def detect_book(image_path, zoom_factor=1.0, apply_clahe=False, apply_sharpening=False,
                detection_threshold=0.1, text_queries=["a book", "open book", "book on table"]):
    """
    Detects books in an image using a Hugging Face zero-shot object detection model.
    Returns a dictionary with "book_detected": True/False.

    Args:
        image_path (str): Path to the input image.
        zoom_factor (float): (Not directly used by Hugging Face pipeline).
        apply_clahe (bool): (Not directly used by Hugging Face pipeline).
        apply_sharpening (bool): (Not directly used by Hugging Face pipeline).
        detection_threshold (float): Minimum confidence score for a detection to be considered valid.
        text_queries (list): List of text descriptions to prompt the model to look for.

    Returns:
        dict: A dictionary containing detection status for "book_detected".
    """
    try:
        # Hugging Face models typically work best with PIL Image objects
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Could not load image from {image_path}")
        return {"book_detected": False}
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return {"book_detected": False}

    # Perform zero-shot object detection
    # The 'candidate_labels' are the text queries the model uses to find objects.
    predictions = hf_detector(image, candidate_labels=text_queries)

    book_detected = False
    
    # You can uncomment this section for debugging to see all detected objects
    # print(f"\n--- Raw Detections for {image_path} ---")
    # for p in predictions:
    #     print(f"Detected: {p['label']}, Score: {p['score']:.2f}, Box: {p['box']}")

    for p in predictions:
        if p['score'] > detection_threshold:
            # We are checking for any detection that meets the confidence threshold
            # based on the provided text queries which are all book-related.
            book_detected = True
            break # Found a book-like object above threshold, no need to check further

    return {
        "book_detected": book_detected
    }


