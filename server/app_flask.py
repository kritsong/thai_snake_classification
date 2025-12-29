from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import json
import os
import io
import base64
import cv2
import cv2
import numpy as np

# Add parent directory to path to allow importing 'common'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.models import get_model
from common.data_manager import get_transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor

app = Flask(__name__)

# --- Configuration ---
USE_HF_TRANSFORMERS = True 
HF_MODEL_ID = "kritaphatson/thai_snake_image_classifier"
placeholder_path = 'models/Swin-Base_augnone_fineT_thres500' 
HF_MODEL_PATH = placeholder_path if os.path.exists(placeholder_path) else HF_MODEL_ID
CHECKPOINT_PATH = 'experiments/EfficientNet-B0_medium_T100/checkpoint_last.pth'
CLASS_INDICES_PATH = 'experiments/EfficientNet-B0_medium_T100/class_indices.json'
MODEL_NAME = 'EfficientNet-B0' 

# Grad-CAM Settings
STAGE_INDEX = 2
AGG_METHOD = "layercam"
ALPHA = 0.40

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_inference_engine():
    if USE_HF_TRANSFORMERS:
        print(f"Loading HF Transformers model from {HF_MODEL_PATH}")
        model = AutoModelForImageClassification.from_pretrained(HF_MODEL_PATH)
        processor = AutoImageProcessor.from_pretrained(HF_MODEL_PATH)
        model.to(device)
        model.eval()
        return model, processor, True
    else:
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)
        model, _, _ = get_model(MODEL_NAME, num_classes, pretrained=False)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        model.to(device)
        model.eval()
        _, val_transform = get_transforms(intensity='none')
        return model, (val_transform, idx_to_class), False

model, engine_data, is_hf = load_inference_engine()

# --- Grad-CAM Logic ---
def get_gradcam(model, pixel_values, target_layer):
    holder = {"acts": None, "grads": None}
    
    def fwd_hook(_m, _inp, out):
        hs = out[0] if isinstance(out, (tuple, list)) else out
        hs.requires_grad_(True)
        hs.retain_grad()
        holder["acts"] = hs
        hs.register_hook(lambda g: holder.__setitem__("grads", g))

    hook = target_layer.register_forward_hook(fwd_hook)
    
    model.zero_grad()
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    idx = torch.argmax(logits, dim=-1)
    score = logits[0, idx]
    score.backward()
    
    hook.remove()
    
    acts = holder["acts"][0].detach().cpu()   # (N, C)
    grads = holder["grads"][0].detach().cpu() # (N, C)
    
    if AGG_METHOD == "layercam":
        cam = torch.relu(grads) * torch.relu(acts)
        cam = cam.sum(dim=-1)
    else: # gradcam
        weights = grads.mean(dim=0)
        cam = torch.relu(acts @ weights)
    
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
        
    # Reshape to 2D
    n = cam.shape[0]
    s = int(np.sqrt(n))
    return cam.reshape(s, s).numpy()

def apply_overlay(orig_img, heatmap, alpha=ALPHA):
    heatmap = cv2.resize(heatmap, (orig_img.width, orig_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    base = np.array(orig_img)
    overlayed = cv2.addWeighted(heatmap, alpha, base, 1 - alpha, 0)
    return Image.fromarray(overlayed)

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Thai Snake Classifier + Grad-CAM</title>
        <style>
            body { font-family: sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; background: #fafafa; }
            .container { background: #fff; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
            h1 { color: #2c3e50; text-align: center; }
            .image-row { display: flex; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap; justify-content: center; }
            .image-box { flex: 1; min-width: 300px; text-align: center; }
            img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #eee; display: none; }
            .result-item { margin: 0.5rem 0; padding: 0.75rem; background: #f8f9fa; border-left: 5px solid #3498db; border-radius: 4px; }
            .confidence-bar { height: 10px; background: #3498db; border-radius: 5px; margin-top: 6px; transition: width 0.5s ease-out; }
            input[type="file"] { display: block; margin: 2rem auto; padding: 10px; border: 2px dashed #ddd; width: 100%; box-sizing: border-box; }
            #status { text-align: center; font-weight: bold; color: #3498db; margin: 1rem 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Thai Snake Classifier</h1>
            <p style="text-align:center">Experimental Swin-Base + Grad-CAM Saliency</p>
            <input type="file" id="imageInput" accept="image/*">
            <div id="status"></div>
            <div class="image-row">
                <div class="image-box">
                    <h3>Input Image</h3>
                    <img id="preview">
                </div>
                <div class="image-box">
                    <h3>Grad-CAM Visualization</h3>
                    <img id="cam_preview">
                </div>
            </div>
            <div id="results" style="margin-top: 2rem;"></div>
        </div>
        <script>
            const imageInput = document.getElementById('imageInput');
            const preview = document.getElementById('preview');
            const camPreview = document.getElementById('cam_preview');
            const resultsDiv = document.getElementById('results');
            const status = document.getElementById('status');

            imageInput.onchange = function(evt) {
                const [file] = imageInput.files;
                if (file) {
                    preview.src = URL.createObjectURL(file);
                    preview.style.display = 'block';
                    camPreview.style.display = 'none';
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    status.innerText = 'Analyzing & Generating Visualization...';
                    resultsDiv.innerHTML = '';
                    
                    fetch('/predict', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        status.innerText = '';
                        camPreview.src = 'data:image/png;base64,' + data.cam_image;
                        camPreview.style.display = 'block';
                        
                        resultsDiv.innerHTML = '<h2>Classification Results:</h2>';
                        data.results.forEach(res => {
                            const div = document.createElement('div');
                            div.className = 'result-item';
                            div.innerHTML = `
                                <strong>${res.species}</strong>: ${(res.confidence * 100).toFixed(2)}%
                                <div class="confidence-bar" style="width: ${res.confidence * 100}%"></div>
                            `;
                            resultsDiv.appendChild(div);
                        });
                    })
                    .catch(error => {
                        status.innerHTML = '<span style="color:red">Error: ' + error.message + '</span>';
                    });
                }
            };
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    raw_img = Image.open(io.BytesIO(file.read())).convert('RGB')
    
    if is_hf:
        processor = engine_data
        inputs = processor(images=raw_img, return_tensors="pt").to(device)
        
        # Determine mapping
        id2label_path = os.path.join(HF_MODEL_PATH, "id2label.json")
        if os.path.exists(id2label_path):
            with open(id2label_path, 'r') as f:
                idx_to_class = json.load(f)
        else:
            idx_to_class = model.config.id2label
        str_idx_to_class = {str(k): v for k, v in idx_to_class.items()}

        # 1. Prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=1)[0]
        
        # 2. Grad-CAM
        target_layer = model.swin.encoder.layers[STAGE_INDEX]
        heatmap = get_gradcam(model, inputs["pixel_values"], target_layer)
        cam_pil = apply_overlay(raw_img, heatmap)
        
    else:
        # TIMM fallback (no CAM implemented yet for general timm)
        val_transform, idx_to_class = engine_data
        str_idx_to_class = {str(k): v for k, v in idx_to_class.items()}
        input_tensor = val_transform(raw_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        cam_pil = raw_img 

    # Encode CAM image to base64
    buffered = io.BytesIO()
    cam_pil.save(buffered, format="PNG")
    cam_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Top Results
    top_prob, top_idx = torch.topk(probabilities, k=min(5, len(str_idx_to_class)))
    
    results = []
    for i in range(len(top_prob)):
        idx_val = top_idx[i].item()
        label = str_idx_to_class.get(str(idx_val), f"Class {idx_val}")
        results.append({'species': label, 'confidence': float(top_prob[i].item())})
        
    return jsonify({
        'results': results,
        'cam_image': cam_base64
    })

if __name__ == '__main__':
    app.run(port=5001, debug=False)
