from flask import Flask, request, Response, send_file
from PIL import Image, ImageDraw ,ImageFont
from ultralytics import YOLO
import io
import json
app = Flask(__name__)

# Assuming you have the best.pt weights file in the same directory as your Flask app
yolov8_weights_path = "best.pt"

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()

@app.route("/upload", methods=["POST"])
def upload():
    """
    Handler of /upload POST endpoint
    Receives uploaded file with a name "image_file", 
    passes it through YOLOv8 object detection 
    network, draws bounding boxes, and returns the modified image.
    :return: Modified image with bounding boxes
    """
    if "image_file" not in request.files:
        return Response(
            json.dumps({"error": "No 'image_file' in the request."}),
            status=400,
            mimetype='application/json'
        )

    buf = request.files["image_file"]
    print(f"Uploaded Image: {buf.filename}")

    image, boxes = detect_objects_on_image(Image.open(buf.stream))
    print(f"Detection Result: {boxes}")

    # Usage in the upload route
    drawn_image = draw_boxes(image, boxes, base_font_size=20)


    # Save the drawn image to a byte stream
    img_byte_array = io.BytesIO()
    drawn_image.save(img_byte_array, format='PNG')
    img_byte_array.seek(0)

    # Return the modified image
    return send_file(img_byte_array, mimetype='image/png')

def detect_objects_on_image(img):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns the modified image and array of detected objects
    and their bounding boxes.
    :param img: Input image
    :return: Modified image, Array of bounding boxes
    """
    model = YOLO(yolov8_weights_path)
    results = model.predict(img ,conf=0.4)
    result = results[0]
    output = []

    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])

    return img, output



def draw_boxes(img, boxes, base_font_size=16):
    """
    Draw bounding boxes on the image with dynamically adjusted font size.
    :param img: Input image
    :param boxes: Array of bounding boxes
    :param base_font_size: Base font size for drawing text (default is 16)
    :return: Image with bounding boxes and annotations drawn
    """
    drawn_image = img.copy()
    draw = ImageDraw.Draw(drawn_image)

    for box in boxes:
        x1, y1, x2, y2, label, _ = box

        # Calculate the dimensions of the bounding box
        box_width = x2 - x1
        box_height = y2 - y1

        # Calculate a dynamic font size based on the bounding box size
        dynamic_font_size = int(base_font_size * min(box_width, box_height) / 50)

        # Create a font with the dynamically calculated size
        font = ImageFont.truetype("arial.ttf", dynamic_font_size)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw label with dynamically adjusted font size
        text = f"{label}"
        draw.text((x1, y1 - 10), text, fill="red", font=font)

    return drawn_image


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
