import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from http.server import SimpleHTTPRequestHandler, HTTPServer
import cgi
import os
from urllib import parse
import json

# Download the model from GitHub
url = 'https://raw.githubusercontent.com/CORNYBUG2/FakevsReal/main/projectthingy.h5'
response = requests.get(url)
with open('projectthingy.h5', 'wb') as f:
    f.write(response.content)

# Load your pre-trained model
model = load_model('projectthingy.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Rescale the values between 0 and 1.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

class MyHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/predict':
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            file_item = form['file']
            if file_item.filename:
                file_path = os.path.join(os.getcwd(), file_item.filename)
                with open(file_path, 'wb') as f:
                    f.write(file_item.file.read())
                preprocessed_image = preprocess_image(file_path)
                prediction = model.predict(preprocessed_image)
                result = 'Fake' if prediction > 0.6 else 'Real'
                os.remove(file_path)  # Clean up the saved file
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"message": result}).encode())
            else:
                self.send_response(400)
                self.end_headers()
        else:
            super().do_GET()

if __name__ == "__main__":
    port = 8000
    server = HTTPServer(('0.0.0.0', port), MyHandler)
    print(f"Starting server on port {port}...")
    server.serve_forever()
