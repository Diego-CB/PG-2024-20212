import cv2
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F

from flask import Flask, request, jsonify
from PIL import Image


# Initialize the model
emotion_dict = {0: "Enojo", 1:"Alegria", 2: "Neutral", 3: "Tristeza", 4: "Sorpresa"}

# --------------------------------------
# resnet
# --------------------------------------
# resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
# resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# resnet.fc = nn.Linear(resnet.fc.in_features, 5)
# model = resnet
# model.load_state_dict(torch.load("./exports/trial2/Resnet_lr001wdNone.pt", weights_only=True))

# --------------------------------------
# VGG
# --------------------------------------
# model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', weights=None)
# model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
# model.classifier[-1] = nn.Linear(4096, 6)
# model.load_state_dict(torch.load("./exports/VGG-11_001.pt"))

# --------------------------------------
# CNn
# --------------------------------------
from src.models.FaceSentiment import ConvNetLarge
model = ConvNetLarge(num_features=5)
model.load_state_dict(torch.load("./exports/trial2/Larger CNN_lr01wdNone.pt", weights_only=True))

# move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Initialize Flask app
app = Flask(__name__)

class HistogramEqualization:
    def __call__(self, img):
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Apply histogram equalization
        img_eq = cv2.equalizeHist(img_array)
        
        # Convert back to PIL image
        return Image.fromarray(img_eq)

preprocess = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(),  # Ensure the image is grayscale
    HistogramEqualization(),
    transforms.ToTensor(),
])

# Function to make predictions
def predict_image(image):
    
    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    # Move the input to the device (CPU or GPU)
    input_tensor = input_tensor.to(device)
    
    # Make the prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # If you are using a softmax output, convert to probabilities
    probabilities = F.softmax(output, dim=1)
    
    # Get the predicted class
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities

# Example usage
import uuid
from datetime import datetime
import os

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    image = Image.open(file)
    predicted_class, _ = predict_image(image)

    return jsonify({"emotion": emotion_dict[predicted_class]})

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
