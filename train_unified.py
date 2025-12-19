import os
import argparse
import json
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Local imports
from data_manager import filter_and_split_data, SnakeDataset, get_transforms, set_seed
from models import get_model, get_lr_config

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / total, 'acc': correct / total})
        
    return running_loss / total, correct / total

def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # AMP for validation too (faster)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser(description="Thai Snake Classification Training")
    parser.add_argument('--data_dir', type=str, default="C:\\Users\\ADMIN\\Downloads\\snake_datasetNew", help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default="experiments", help='Dir to save results')
    parser.add_argument('--threshold', type=int, default=100, choices=[100, 200, 300, 400, 500])
    parser.add_argument('--model', type=str, required=True, choices=['MobileNetV3-Small', 'EfficientNet-B0', 'ResNet-50', 'ViT-Base/16', 'Swin-Base'])
    parser.add_argument('--aug_intensity', type=str, default='medium', choices=['none', 'low', 'medium', 'high'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 0. Setup
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    exp_name = f"{args.model}_{args.aug_intensity}_T{args.threshold}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"--- Starting Experiment: {exp_name} ---")
    
    # 1. Data
    print("Preparing Data...")
    splits = filter_and_split_data(args.data_dir, args.threshold, seed=args.seed)
    if splits[0] is None:
        print(f"No classes meet threshold {args.threshold}. Exiting.")
        return

    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_to_idx = splits
    print(f"Classes: {len(class_to_idx)}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Save class mapping
    with open(os.path.join(exp_dir, 'class_indices.json'), 'w') as f:
        json.dump(class_to_idx, f, indent=4)
        
    # Transforms
    train_tf, val_tf = get_transforms(args.aug_intensity)
    
    # Datasets
    train_set = SnakeDataset(X_train, y_train, transform=train_tf)
    val_set = SnakeDataset(X_val, y_val, transform=val_tf)
    test_set = SnakeDataset(X_test, y_test, transform=val_tf)
    
    # Loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    # Test loader can be used for final eval
    
    # 2. Model & Optimizer
    print("Creating Model...")
    model, backbone_params, head_params = get_model(args.model, len(class_to_idx))
    model = model.to(device)
    
    backbone_lr, head_lr = get_lr_config(args.model)
    
    # Initialize Optimizer
    # Note: "During the first 10 epochs, the backbone remained frozen and only the classification head was updated."
    # Strategy:
    # 1. We construct the optimizer with BOTH groups, but we can set dataset requires_grad or just use the optimizer.
    # If we want to freeze, setting `requires_grad=False` is safest.
    # We will initialize optimizer with both groups so we don't have to re-init it later, 
    # BUT if gradients are 0 (frozen), it doesn't matter.
    # Actually, AdamW maintains state. If we add params later, state is missing.
    # So best to add all, but freeze grads.
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': head_params, 'lr': head_lr}
    ], weight_decay=0.05)
    
    # Scheduler: Cosine over FULL horizon (50 epochs)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # 3. Training Loop
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        # Freeze/Unfreeze Logic
        if epoch < 10:
            # Frozen Backbone
            for p in backbone_params:
                p.requires_grad = False
            for p in head_params:
                p.requires_grad = True
            phase_status = "Backbone Frozen"
        else:
            # Unfrozen
            for p in backbone_params:
                p.requires_grad = True
            for p in head_params:
                p.requires_grad = True
            phase_status = "Full Training"
            
        print(f"\nEpoch {epoch+1}/{args.epochs} - {phase_status} - LR: {[g['lr'] for g in optimizer.param_groups]}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
        
        # Step scheduler
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save checkpoint (last)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, os.path.join(exp_dir, 'checkpoint_last.pth'))
        
    # 4. Final Evaluation on Test Set
    print("\n--- Final Evaluation on Test Set ---")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.num_workers, pin_memory=True)
    
    test_loss, test_acc = validate(model, test_loader, criterion, device, args.epochs)
    print(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")
    
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    
    # Save Final Results
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
        
    print("Training and Evaluation Complete.")

if __name__ == '__main__':
    main()
