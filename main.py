from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok

# Load the model
model = load_model('https://raw.githubusercontent.com/CORNYBUG2/FakevsReal/refs/heads/main/projectthingy.h5')

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET'])
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Deepfake Detector</title>
    </head>
    <body>
        <h1>Deepfake Detector</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    '''

@app.route('/', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'})

    file = request.files['image']
    image = Image.open(file)
    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)
    result = 'Real' if prediction[0][0] > 0.5 else 'Fake'

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run()
