import torch
import torch.nn as nn
import timm

def get_model(model_name, num_classes, pretrained=True):
    """
    Creates a model from timm and returns it along with parameter groups for optimization.
    
    Models requested:
    - MobileNetV3-Small: 'mobilenetv3_small_100'
    - EfficientNet-B0: 'efficientnet_b0'
    - ResNet-50: 'resnet50'
    - ViT-Base/16: 'vit_base_patch16_224'
    - Swin-Base: 'swin_base_patch4_window7_224'
    
    Returns:
        model: The PyTorch model.
        param_groups: List of dicts for optimizer [{'params': backbone, 'lr': base_lr}, {'params': head, 'lr': head_lr}]
                      (Note: The actual LRs will be set in the optimizer, this function just separates the params).
    """
    
    # Map friendly names to timm names
    name_map = {
        'MobileNetV3-Small': 'mobilenetv3_small_100',
        'EfficientNet-B0': 'efficientnet_b0',
        'ResNet-50': 'resnet50',
        'ViT-Base/16': 'vit_base_patch16_224',
        'Swin-Base': 'swin_base_patch4_window7_224'
    }
    
    timm_name = name_map.get(model_name, model_name)
    
    print(f"Creating model: {timm_name}")
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    
    # Identify head parameters to separate them from backbone
    # timm models usually have 'head', 'fc', or 'classifier'.
    # We can use model.get_classifier() to find the module, but we need the names/parameters.
    
    head_names = []
    # Common head names in timm
    potential_heads = ['classifier', 'head', 'fc']
    
    found_head = False
    for h in potential_heads:
        if hasattr(model, h):
            # Check if it's a module
            mod = getattr(model, h)
            if isinstance(mod, nn.Module):
                # This is likely the head
                # Get all parameter names start with this prefix
                head_names.append(h)
                found_head = True
                break
    
    if not found_head:
        # Fallback: Print warning, treat all as backbone (should not happen for these standard models)
        print(f"WARNING: Could not identify classification head for {model_name}. Treating all as backbone.")
        backbone_params = list(model.parameters())
        head_params = []
    else:
        # Separate params
        head_params = []
        backbone_params = []
        
        head_prefix = head_names[0]
        
        for name, param in model.named_parameters():
            if name.startswith(head_prefix):
                head_params.append(param)
            else:
                backbone_params.append(param)
                
    return model, backbone_params, head_params

def get_lr_config(model_name):
    """
    Returns specific LR settings based on model family (CNN vs Transformer).
    
    From prompt:
    Transformers: Backbone 2e-5, Head 2e-4
    CNNs: Backbone 1e-4, Head 1e-3
    """
    transformers = ['ViT-Base/16', 'Swin-Base']
    
    if model_name in transformers:
        return 2e-5, 2e-4
    else:
        # CNNs
        return 1e-4, 1e-3
