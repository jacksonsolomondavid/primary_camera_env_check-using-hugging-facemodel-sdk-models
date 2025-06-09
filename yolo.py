import cv2
import numpy as np
from ultralytics import YOLO

# --- Preprocessing Function ---
def preprocess_image(image, zoom_factor=1.5, apply_clahe=False, apply_sharpening=False):
    """
    Applies various preprocessing steps to an image.

    Args:
        image (np.array): The input image (OpenCV format).
        zoom_factor (float): Factor to resize the image (e.g., 1.5 for 50% larger).
                             YOLO will internally resize further to its input size.
        apply_clahe (bool): Whether to apply Contrast Limited Adaptive Histogram Equalization.
        apply_sharpening (bool): Whether to apply a simple sharpening filter.

    Returns:
        np.array: The preprocessed image.
    """
    processed_image = image.copy()

    # 1. Zoom (Resizing)
    if zoom_factor != 1.0:
        new_width = int(processed_image.shape[1] * zoom_factor)
        new_height = int(processed_image.shape[0] * zoom_factor)
        processed_image = cv2.resize(processed_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        print(f"Image zoomed by {zoom_factor}x to {processed_image.shape[1]}x{processed_image.shape[0]} pixels.")

    # Convert to grayscale for some operations if needed, or process channels separately
    # For CLAHE and Sharpening on color images, we'll process channels.
    
    if apply_clahe:
        # Apply CLAHE to each channel (or convert to LAB/HSV for better results)
        lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(l_channel)
        lab[:, :, 0] = cl1
        processed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        print("Applied CLAHE.")

    if apply_sharpening:
        # Simple sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        processed_image = cv2.filter2D(processed_image, -1, kernel)
        print("Applied sharpening.")

    return processed_image

# --- Main Script ---

# Load the LARGER YOLO model for this test
# MAKE SURE YOU CHANGE THIS LINE to "yolov8s.pt" or "yolov8m.pt"
test_model = YOLO("yolov8s.pt") # <--- Change this to your chosen larger model

image_path = "book1.jpg" # <--- Use the earphones.jpg image

# Read the original image
original_img = cv2.imread(image_path)

if original_img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    model_name_display = test_model.model_name if hasattr(test_model, 'model_name') else "YOLO Model"
    print(f"Processing original image: {image_path} with {model_name_display}...")

    # --- Apply Preprocessing ---
    # Experiment with these parameters
    processed_img = preprocess_image(original_img, 
                                     zoom_factor=1.5,      # Try 1.2, 1.5, 2.0 to zoom in
                                     apply_clahe=True,     # Try True/False
                                     apply_sharpening=False) # Try True/False

    print(f"Running inference on preprocessed image (shape: {processed_img.shape}).")

    # Perform inference on the PREPROCESSED image
    results = test_model(processed_img)[0]

    # Print all detected objects and their labels
    print("\n--- Raw YOLO Detections (from preprocessed image) ---")
    found_earphone_label = False
    for r in results.boxes.data.tolist():
        if len(r) >= 6:
            x1, y1, x2, y2, conf, cls_id = r
            label = results.names.get(int(cls_id), "unknown").lower()
            print(f"Detected: {label}, Confidence: {conf:.2f}, Bounding Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
            
            if "earphone" in label or "headphone" in label:
                found_earphone_label = True

    if found_earphone_label:
        print("\n--- YOLO model DID detect an earphone/headphone label directly. ---")
    else:
        print("\n--- YOLO model did NOT detect any earphone/headphone labels directly. ---")

    # The original image (original_img) is still available here if needed for other purposes.
    # For example, you could visualize original_img and then overlay bounding boxes from results
    # that were generated using processed_img.
    # cv2.imshow("Original Image", original_img)
    # cv2.imshow("Preprocessed Image", processed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()