import torch
import torch.nn.functional as F
from PIL import Image
import json
import argparse
import os
# Add parent directory to path to allow importing 'common'
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.models import get_model
from common.data_manager import get_transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor

def run_inference(image_path, model_path, class_indices_path, model_name, device, top_k=5):
    # Determine if it's a transformers model or a custom timm model
    is_hf_transformers = False
    if os.path.isdir(model_path):
        if os.path.exists(os.path.join(model_path, "config.json")):
            is_hf_transformers = True

    if is_hf_transformers:
        print(f"Loading Hugging Face Transformers model from: {model_path}")
        model = AutoModelForImageClassification.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
        model.to(device)
        model.eval()
        
        # Mapping from config
        idx_to_class = model.config.id2label
        num_classes = len(idx_to_class)
    else:
        # Load class indices
        if class_indices_path is None or not os.path.exists(class_indices_path):
            raise ValueError(f"Class indices path must be provided for .pth models. Path given: {class_indices_path}")
            
        with open(class_indices_path, 'r') as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)

        # Load model
        print(f"Loading timm model: {model_name}")
        model, _, _ = get_model(model_name, num_classes, pretrained=False)
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        _, val_transform = get_transforms(intensity='none')
        processor = lambda x: val_transform(x).unsqueeze(0).to(device)

    # Preprocess image
    image = Image.open(image_path).convert('RGB')
    
    if is_hf_transformers:
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=1)[0]
    else:
        input_tensor = processor(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
    # Get top-K
    top_prob, top_idx = torch.topk(probabilities, k=min(top_k, num_classes))
    
    results = []
    for i in range(len(top_prob)):
        label = idx_to_class[str(top_idx[i].item())] if isinstance(idx_to_class, dict) and str(top_idx[i].item()) in idx_to_class else idx_to_class[top_idx[i].item()]
        results.append({
            "species": label,
            "confidence": top_prob[i].item()
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Local Inference for Thai Snake Classification")
    parser.add_argument('--image', type=str, required=True, help='Path to image file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth checkpoint OR HF model directory')
    parser.add_argument('--class_indices', type=str, default=None, help='Path to class_indices.json (required for .pth models)')
    parser.add_argument('--model_name', type=str, default='EfficientNet-B0', help='Model architecture name (for timm models)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top classes to show')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    results = run_inference(
        args.image, 
        args.model_path, 
        args.class_indices, 
        args.model_name, 
        device, 
        args.top_k
    )
    
    print(f"\nPredictions for {os.path.basename(args.image)}:")
    for res in results:
        print(f"  {res['species']}: {res['confidence']*100:.2f}%")

if __name__ == "__main__":
    main()
