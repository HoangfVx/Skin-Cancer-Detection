import numpy as np
import torch
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, labels in dataloader:
            images = images.to(device)  # Move images to GPU
            labels = labels.float().to(device)  # Move labels to GPU

            outputs = model(images)  # Get model predictions
            preds = (outputs.squeeze() > 0.5).float()  # Apply sigmoid and threshold
            
            all_labels.extend(labels.cpu().numpy())  # Collect true labels
            all_preds.extend(preds.cpu().numpy())  # Collect predicted labels
    return np.array(all_labels), np.array(all_preds)

