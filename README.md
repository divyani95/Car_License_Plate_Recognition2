# Car_License_Plate_Recognition
This project is a Number Plate Recognition and Extraction system that uses advanced computer vision and machine learning techniques to detect and recognize license plates from video footage. It leverages the YOLO (You Only Look Once) model for vehicle and license plate detection, the SORT (Simple Online and Realtime Tracking) algorithm for vehicle tracking, and OpenCV for image processing. The recognized license plate numbers are saved in a CSV file for further analysis.

Features
Vehicle Detection: Detects vehicles such as cars, buses, trucks, and motorbikes using the YOLOv8 model.

License Plate Detection: Uses a custom-trained YOLO model to detect license plates on vehicles.

License Plate Recognition: Extracts text from detected license plates using image processing techniques.

Vehicle Tracking: Tracks vehicles across frames using the SORT algorithm.

Real-Time Visualization: Displays the live video feed with bounding boxes around vehicles and license plates, along with the recognized license plate text.

CSV Output: Saves the detected license plate numbers, bounding box coordinates, and confidence scores in a CSV file.

Prerequisites
Before running the project, ensure you have the following installed:

Python 3.8 or higher

OpenCV (opencv-python)

Ultralytics YOLO (ultralytics)

NumPy (numpy)

Screeninfo (screeninfo)

SORT (Simple Online and Realtime Tracking) - Included in the project as sort.py

You can install the required Python packages using the following command:

bash
Copy
pip install opencv-python ultralytics numpy screeninfo
Project Structure
The project directory is organized as follows:

Copy
Number-plate-Recognition-and-Extraction/
│
├── sort.py                  # SORT tracking algorithm
├── util.py                  # Utility functions (get_car, read_license_plate, write_csv)
├── yolov8n.pt               # Pretrained YOLOv8 model for vehicle detection
├── license_plate_detector.pt # Custom-trained YOLO model for license plate detection
├── sample.mp4               # Sample video for testing
├── test.csv                 # Output CSV file with detected license plate data
└── Number_Recognition.py    # Main script for number plate recognition
How to Run the Project
Clone the Repository:

bash
Copy
git clone https://github.com/divyani95/Car_License_Plate_Recognition2/blob/main/Number_Recognition.py
cd Number-plate-Recognition-and-Extraction
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Run the Script:

bash
Copy
python Number_Recognition.py
View the Output:

The script will display the live video feed with bounding boxes around vehicles and license plates.

Detected license plate numbers will be saved in test.csv.

Customization
Video Input: Replace sample.mp4 with your own video file by updating the path in the script:

python
Copy
cap = cv2.VideoCapture(r"path/to/your/video.mp4")
Confidence Threshold: Adjust the confidence threshold for license plate detection:

python
Copy
license_plates = license_plate_detector(frame, conf=0.3)[0]  # Lower confidence threshold
Skip Frames: Modify the number of frames to skip for faster processing:

python
Copy
skip_frames = 2  # Process every 2nd frame
Output
The script generates the following outputs:

Live Video Feed:

Displays the video with bounding boxes around vehicles (blue) and license plates (green).

Recognized license plate text is displayed in black on a white background.

CSV File (test.csv):

Contains the following columns for each detected license plate:

frame_nmr: Frame number in the video.

car_id: Unique ID of the tracked vehicle.

car_bbox: Bounding box coordinates of the vehicle.

license_plate_bbox: Bounding box coordinates of the license plate.

license_plate_text: Recognized license plate number.

bbox_score: Confidence score of the license plate detection.

text_score: Confidence score of the text recognition.

Applications
This project can be used in various real-world applications, including:

Traffic Monitoring: Automate vehicle identification for traffic management.

Law Enforcement: Detect and track vehicles of interest.

Automated Toll Systems: Streamline toll collection using license plate recognition.

Parking Management: Monitor vehicles entering and exiting parking facilities.

Fleet Tracking: Track and manage logistics fleets.

Limitations
Accuracy: The accuracy of license plate recognition depends on the quality of the video and the training data used for the YOLO models.

Processing Speed: The script may run slower on low-end hardware. Consider skipping more frames or reducing the video resolution for faster processing.

Lighting Conditions: Performance may degrade in poor lighting conditions.

Future Improvements
GPU Acceleration: Use GPU for faster inference with YOLO models.

Better OCR: Integrate more advanced OCR (Optical Character Recognition) techniques for improved text recognition.

Multiple License Plates: Extend the system to handle vehicles with multiple license plates (e.g., front and rear).

Real-Time Deployment: Deploy the system on edge devices for real-time applications.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
YOLOv8: For the state-of-the-art object detection model.

SORT: For the efficient object tracking algorithm.

OpenCV: For image processing and visualization.



