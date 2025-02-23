import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Import from the local sort.py file
from util import get_car, read_license_plate, write_csv
from screeninfo import get_monitors  # To get screen resolution

results = {}

mot_tracker = Sort()

# Load models (CPU only)
coco_model = YOLO('yolov8n.pt')  # Use CPU
license_plate_detector = YOLO('./license_plate_detector.pt')  # Use CPU

cap = cv2.VideoCapture(r"C:\Users\DELL\Downloads\AVAADA\Number-plate-Recognition-and-Extraction\sample.mp4")
if not cap.isOpened():
    raise Exception("Could not open video file.")

# Get screen resolution
monitor = get_monitors()[0]  # Get the primary monitor
screen_width = monitor.width
screen_height = monitor.height

# Define vehicle classes (COCO dataset)
vehicles = [2, 3, 5, 7]  # 2: car, 3: motorbike, 5: bus, 7: truck

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to improve speed
    skip_frames = 2  # Process every 2nd frame
    if frame_nmr % skip_frames != 0:
        continue

    results[frame_nmr] = {}

    # Detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Detect license plates with a lower confidence threshold
    license_plates = license_plate_detector(frame, conf=0.3)[0]  # Lower confidence threshold
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # Crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # Process license plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Read license plate number
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score
                    }
                }

                # Draw bounding box and text on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box for license plate

                # Draw white background for text
                (text_width, text_height), _ = cv2.getTextSize(license_plate_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10), (int(x1) + text_width, int(y1)), (255, 255, 255), -1)

                # Draw black text on white background
                cv2.putText(frame, license_plate_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 2)

                # Draw bounding box for the car
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (255, 0, 0), 2)  # Blue box for car
                cv2.putText(frame, f"Car ID: {car_id}", (int(xcar1), int(ycar1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Resize frame to fit screen while maintaining aspect ratio
    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height

    # Calculate new dimensions to fit the screen
    if frame_width > screen_width or frame_height > screen_height:
        if aspect_ratio > (screen_width / screen_height):
            # Width is the limiting factor
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)
        else:
            # Height is the limiting factor
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)
    else:
        # Frame is smaller than the screen, no need to resize
        new_width, new_height = frame_width, frame_height

    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Display the resized frame
    cv2.imshow("Live License Plate Detection", resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save results to CSV
write_csv(results, './test.csv')

print("Live detection completed. Results saved to 'test.csv'.")