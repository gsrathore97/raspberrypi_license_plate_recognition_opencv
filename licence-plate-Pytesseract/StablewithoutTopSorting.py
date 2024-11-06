import cv2
import numpy as np
from picamera2 import Picamera2
from PlateExtraction import extraction
from OpticalCharacterRecognition import ocr, check_if_string_in_file
from datetime import datetime, timedelta
import os

# Initialize Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Create directories for logging plates and saving images
log_file_path = './Database/detected_plates_log.txt'
output_directory = './Database/detected_plates_images'
os.makedirs(output_directory, exist_ok=True)

# Dictionary to store the last detected time for each plate
last_detected_times = {}

while True:
    # Capture frame from Picamera2
    frame = picam2.capture_array()
    
    # Extract the license plate from the frame
    plate = extraction(frame)
    
    # Check if 'plate' is a valid image before processing
    if plate is not None and isinstance(plate, (np.ndarray,)):
        try:
            # Perform OCR on the extracted plate
            text = ocr(plate)
            text = ''.join(e for e in text if e.isalnum())
        except Exception as e:
            print(f"OCR processing error: {e}")
            continue  # Skip to the next frame if there's an error in OCR
    else:
        print("Extraction error: No valid plate detected")
        continue

    if text != '':
        print(text, end=" ")
        # Check if the plate is registered
        is_registered = check_if_string_in_file('./Database/Database.txt', text)
        
        # Prepare status message
        status = 'Registered' if is_registered else 'Not Registered'
        print(status)

        # Get the current time
        current_time = datetime.now()

        # Check if the plate has been logged within the last minute
        if text in last_detected_times:
            last_time = last_detected_times[text]
            if current_time - last_time < timedelta(minutes=1):
                print("Duplicate entry, skipping log and save.")
                continue  # Skip logging and saving if within 1 minute

        # Log the detected plate with date and time
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"{current_time.strftime('%Y-%m-%d %H:%M:%S')} | Plate: {text} | Status: {status}\n")
        
        # Save the cropped plate image with detected number, date, and time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f"plate_{text}_{timestamp}.jpg"
        image_path = os.path.join(output_directory, image_filename)
        cv2.imwrite(image_path, plate)  # Save the plate image

        # Update the last detected time for this plate
        last_detected_times[text] = current_time

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
picam2.stop()
