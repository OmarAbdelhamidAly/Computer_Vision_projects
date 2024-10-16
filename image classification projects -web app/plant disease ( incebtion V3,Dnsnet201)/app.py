import os
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models = {}

ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'jfif'}
classes = {
   'Bell pepper': ['Bell pepper Bacterial spot', 'Bell pepper Healthy'],
    'Cherry': ['Cherry Healthy', 'Cherry Powdery mildew'],
    'Cirus': ['Citrus Black spot', 'Citrus canker', 'Citrus greening', 'Citrus Healthy'],
    'Corn': ['Corn Common rust', 'Corn Gray leaf spot', 'Corn Healthy', 'Corn Northern Leaf Blight'],
    'Grape': ['Grape Black Measles', 'Grape Black rot', 'Grape Healthy', 'Grape Isariopsis Leaf Spot'],
    'Peach': ['Peach Bacterial spot', 'Peach Healthy'],
    'Strawberry': ['Strawberry Healthy', 'Strawberry Leaf scorch'],
    'Apple': ['Apple Black rot', 'Apple Healthy', 'Apple Scab', 'Cedar apple rust'],
    'Potato': ['Potato Early blight', 'Potato Healthy', 'Potato Late blight'],
    'tomato': ['Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Healthy', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Mosaic virus', 'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus']
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def load_local_models():
    model_files = {
    'Bell pepper':'Bell pepper.h5',
    'Cherry' : 'Cherry.h5',
    'Cirus': 'Citrus.h5',
    'Corn': 'Corn.h5',
    'Grape':'Grape.h5',
    'Oeach': 'Peach.h5',
    'Strawberry': 'Strawberry.h5',
    'Apple':'Apple.h5',
    'Potato': 'Potato.h5',
     'Tomato':'Tomato.h5'
    }

    for model_name, model_file in model_files.items():
        model_path = os.path.join(BASE_DIR, model_file)
        model = load_model(model_path)
        models[model_name] = model


def predict(filename, model, model_name):
    img = load_img(filename, target_size=(256, 256))
    img = img_to_array(img)
    img = img.reshape(1, 256, 256, 3)
    img = img.astype('float32') / 255.0

    result = model.predict(img)[0]
    sorted_indices = (-result).argsort()

    predictions = []
    for i in sorted_indices[:3]:
        class_name = classes[model_name][i]
        probability = round(result[i] * 100, 3)
        predictions.append({'class': class_name, 'probability': probability})

    return predictions


@app.route('/')
def home():
        return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict_image():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file found in the request'})

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    model_name = request.args.get('model')
    if not model_name:
        return jsonify({'error': 'No model specified'})

    if model_name not in models:
        return jsonify({'error': 'Invalid model specified'})

    if file and allowed_file(file.filename):
        # Generate a unique filename based on the original name and format
        original_filename = secure_filename(file.filename)
        img_path = os.path.join(BASE_DIR, 'static/images', original_filename)

       # Check if the file already exists, and if so, return the existing file's predictions
        if os.path.exists(img_path):
            class_result = predict(img_path, models[model_name], model_name)
        else:
            file.save(img_path)
            class_result = predict(img_path, models[model_name], model_name)
        predictions = {
            "class1": class_result[0]['class'],
            "class2": class_result[1]['class'],
            "class3": class_result[1]['class'],
            "prob1": class_result[0]['probability'],
            "prob2": class_result[1]['probability'],
            "prob3": class_result[1]['probability'],
        }
        response = jsonify(predictions)
        
        return response
    else:
        return jsonify({'error': 'Invalid file format'})



if __name__ == "__main__":
    # Load models locally
    load_local_models()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)),
            debug=False)
