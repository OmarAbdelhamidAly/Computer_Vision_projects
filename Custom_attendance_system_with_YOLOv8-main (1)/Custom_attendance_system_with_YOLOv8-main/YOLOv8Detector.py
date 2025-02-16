from ultralytics import YOLO
from flask import request, Response, Flask, render_template, jsonify
from PIL import Image
import base64
import json

app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        user_id = data.get('id')
        image_data = data.get('image_data')

        buf = Image.open(base64.b64decode(image_data.split(',')[1]))
        boxes = detect_objects_on_image(buf)

        if any("Mohammed Dewedar" in obj[4] for obj in boxes):
            result = "Mohammed"
        else:
            result = None

        return jsonify({'result': result})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'An error occurred. Please try again.'}), 500

def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
            round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
            x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

if __name__ == "__main__":
    app.run(host='0.0.0.0')
