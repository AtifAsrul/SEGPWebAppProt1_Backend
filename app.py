from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

app = Flask(__name__)

# 🧠 Load pretrained model
weights = MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights)
model.eval()

# 🧾 Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
categories = weights.meta["categories"]

@app.route('/')
def home():
    return "✅ Flask backend with MobileNetV2 is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')

    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_catid = torch.topk(probs, 1)

    return jsonify({
        "label": categories[top_catid[0]],
        "confidence": float(top_prob[0])
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
