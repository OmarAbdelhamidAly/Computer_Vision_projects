from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import os
app = Flask(__name__)

# Replace this dictionary with your actual mapping of user IDs to known faces
user_id_to_face_mapping = {
    '205171': 'Mohammed Dewedar',
}

class YOLOv8Detector:
    def __init__(self, weights_path, config_path, names_path):
        self.net = self.load_model(weights_path, config_path)
        self.classes = self.load_classes(names_path)

    def load_model(self, weights_path, config_path):
        # Load YOLOv8 model using cv2.dnn.readNetFromDarknet
        try:
            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        except cv2.error as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise  # Re-raise the exception to see the full traceback
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def load_classes(self, file_path):
        with open(file_path, 'r') as f:
            classes = f.read().strip().split('\n')
        return classes

    def detect_objects(self, image):
        height, width, _ = image.shape

        # YOLOv8 architecture specific configuration
        layer_names = self.net.getUnconnectedOutLayersNames()
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(layer_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Adjust confidence threshold as needed
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        class_names = [self.classes[class_id] for class_id in class_ids]

        # Collecting results
        results = []
        for box, confidence, class_id, class_name in zip(boxes, confidences, class_ids, class_names):
            result = {
                'box': box,
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            }
            results.append(result)

        return results

    def detect_face(self, user_id, image_path):
        # Placeholder method for face detection based on user_id
        # Modify this method according to your actual implementation
        # Return True if the face is detected, False otherwise
        image = cv2.imread(image_path)
        detection_results = self.detect_objects(image)

        # Assuming that "Mohammed Dewedar" class ID is 0 (modify if needed)
        is_detected = any(result['class_id'] == 0 for result in detection_results)
        return is_detected


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_form', methods=['GET', 'POST'])
def detect_form():
    if request.method == 'POST':
        # Get the entered ID from the form
        user_id = request.form.get('id')

        # Check if the entered ID is valid
        if user_id in user_id_to_face_mapping:
            yolo_detector = YOLOv8Detector("pest.pt", "data_custom.yaml", "classes")

            # Get the base64-encoded image data from the form
            captured_image_data = request.form.get('capturedImage')

            # Convert base64 to binary and save to a temporary location
            temp_image_path = os.path.join("temp", "captured_image.jpg")
            with open(temp_image_path, 'wb') as f:
                f.write(captured_image_data.split(',')[1].decode('base64'))

            # Perform face detection on the captured image
            detection_result = yolo_detector.detect_face(user_id, temp_image_path)

            if detection_result:
                return render_template('detection_result.html', message=f'Hi {user_id_to_face_mapping[user_id]}!')
            else:
                return render_template('detection_result.html', message='Face not detected. Please try again.')

        return render_template('detect_form.html', error='Invalid ID. Please try again.')

    # If it's a GET request, render the form
    return render_template('detect_form.html', error=None)

if __name__ == '__main__':
    app.run(debug=True)