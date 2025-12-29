import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import json
import os
import io
import base64
import cv2
import numpy as np
# Add parent directory to path to allow importing 'common'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.models import get_model
from common.data_manager import get_transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor

# --- Configuration ---
# You can set these to local paths or Hugging Face repo IDs
USE_HF_TRANSFORMERS = True 
HF_MODEL_ID = "kritaphatson/thai_snake_image_classifier" # Use repo ID for Space deployment

# For local testing, you can override with a local path
LOCAL_MODEL_PATH = 'models/Swin-Base_augnone_fineT_thres500' # Placeholder for local path
model_path = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else HF_MODEL_ID

# Grad-CAM Settings
STAGE_INDEX = 2
AGG_METHOD = "layercam"
ALPHA = 0.40

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_inference_engine():
    print(f"Loading HF Transformers model from {model_path}")
    model = AutoModelForImageClassification.from_pretrained(model_path)
    processor = AutoImageProcessor.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Try to load id2label
    if hasattr(model.config, 'id2label'):
        idx_to_class = model.config.id2label
    else:
        # Fallback to local file if exists
        id2label_path = os.path.join(model_path, "id2label.json") if os.path.isdir(model_path) else None
        if id2label_path and os.path.exists(id2label_path):
            with open(id2label_path, 'r') as f:
                idx_to_class = json.load(f)
        else:
            idx_to_class = {i: f"Class {i}" for i in range(model.config.num_labels)}
            
    return model, processor, idx_to_class

model, processor, idx_to_class = load_inference_engine()
str_idx_to_class = {str(k): v for k, v in idx_to_class.items()}

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
    heatmap = cv2.resize(heatmap, (orig_img.size[0], orig_img.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    base = np.array(orig_img)
    overlayed = cv2.addWeighted(heatmap, alpha, base, 1 - alpha, 0)
    return Image.fromarray(overlayed)

def predict(image):
    if image is None: return None
    
    raw_img = Image.fromarray(image).convert('RGB')
    inputs = processor(images=raw_img, return_tensors="pt").to(device)
    
    # 1. Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)[0]
    
    # 2. Grad-CAM
    target_layer = model.swin.encoder.layers[STAGE_INDEX]
    heatmap = get_gradcam(model, inputs["pixel_values"], target_layer)
    cam_pil = apply_overlay(raw_img, heatmap)
    
    # Top Results
    top_prob, top_idx = torch.topk(probabilities, k=min(5, len(str_idx_to_class)))
    
    confidences = {}
    for i in range(len(top_prob)):
        idx_val = top_idx[i].item()
        label = str_idx_to_class.get(str(idx_val), f"Class {idx_val}")
        confidences[label] = float(top_prob[i].item())
        
    return cam_pil, confidences

# --- Gradio Interface ---
with gr.Blocks(title="Thai Snake Classifier + Grad-CAM") as demo:
    gr.Markdown("# Thai Snake Classifier")
    gr.Markdown("Model: Swin-Base with Grad-CAM visualization for interpretability.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Snake Image")
            btn = gr.Button("Classify")
        with gr.Column():
            output_cam = gr.Image(label="Grad-CAM Visualization")
            output_labels = gr.Label(num_top_classes=5)
            
    btn.click(fn=predict, inputs=input_img, outputs=[output_cam, output_labels])

if __name__ == "__main__":
    demo.launch()
