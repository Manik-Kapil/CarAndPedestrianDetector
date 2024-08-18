# from flask import Flask, request, render_template, jsonify
# import torch
# from PIL import Image
# import torchvision.transforms as T
# import io
# import base64
# import matplotlib.pyplot as plt
# import torchvision
# from predictor import Predictor  # Import the Predictor class
# import logging

# app = Flask(__name__)

# # Load the trained model
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# num_classes = 3  # Adjust based on your number of classes (2 categories + background)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
# try:
#     model.load_state_dict(torch.load('trained_model.pth', map_location=device), strict=False)
#     print("Model loaded successfully")
# except RuntimeError as e:
#     print(f"Error loading model state_dict: {e}")
# model.to(device)
# model.eval()

# # Create an instance of Predictor
# predictor = Predictor(model)

# def load_image(image):
#     transform = T.Compose([T.ToTensor()])
#     image_tensor = transform(image).unsqueeze(0)
#     print(f"Image Tensor Shape: {image_tensor.shape}")  # Debug tensor shape
#     return image_tensor

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     logging.debug("Index route triggered")
#     if request.method == 'POST':
#         logging.debug("POST request received")
#         file = request.files['file']
#         if file:
#             logging.debug("File received")
#             image = Image.open(file.stream).convert("RGB")
#             image_tensor = load_image(image).to(device)
#             predictions = predictor.predict(image_tensor)
#             logging.debug(f"Predictions: {predictions}")

#             # Plot and save the image with bounding boxes
#             fig, ax = plt.subplots(1)
#             ax.imshow(image)

#             # Process and plot predictions
#             if predictions:
#                 for element in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
#                     box, label, score = element
#                     if score > 0.5:  # Confidence threshold
#                         x1, y1, x2, y2 = box
#                         ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', fill=False, linewidth=2))
#                         ax.text(x1, y1, f'{label}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

#                 img_io = io.BytesIO()
#                 plt.savefig(img_io, format='PNG')
#                 img_io.seek(0)
#                 img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')
#                 return img_base64
#             else:
#                 return "No predictions found."

#     return render_template('index.html')

# if __name__ == "__main__":
#     app.run(debug=True, port=50001)



from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
import torchvision.transforms as T
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision

app = Flask(__name__)

# Load the trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 3  # Adjust based on your number of classes (2 categories + background)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.to(device)
model.eval()

def load_image(image):
    transform = T.Compose([T.ToTensor()])
    image = transform(image).unsqueeze(0)
    return image

def get_prediction(image):
    with torch.no_grad():
        prediction = model(image)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream).convert("RGB")
            image_tensor = load_image(image).to(device)
            predictions = get_prediction(image_tensor)

            # Plot and save the image with bounding boxes
            fig, ax = plt.subplots(1)
            ax.imshow(image)

            for element in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
                box, label, score = element
                if score > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box
                    ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', fill=False, linewidth=2))
                    ax.text(x1, y1, f'{label}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))

            img_io = io.BytesIO()
            plt.savefig(img_io, format='PNG')
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')
            return img_base64

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
