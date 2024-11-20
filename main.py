import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
import requests

app = FastAPI()

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download the model from GitHub
thisthing = 'https://raw.githubusercontent.com/CORNYBUG2/FakevsReal/refs/heads/main/projectthingy.h5'

response = requests.get(thisthing)
open('projectthingy.h5', 'wb').write(response.content)

# Load your pre-trained model
model = load_model('projectthingy.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    img = Image.open(file.file).convert('RGB')
    img = img.resize((224, 224))  # Adjust this size based on your model's requirements
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    result = 'Fake' if prediction > 0.5 else 'Real'
    return {"message": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
