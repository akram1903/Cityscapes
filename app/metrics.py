import torch
from output_processing import process_output
from tqdm import tqdm
import numpy as np
def pixel_accuracy(pred, target):
    """
    Compute pixel accuracy for a batch of segmentation predictions.

    Args:
        pred (torch.Tensor): The predicted segmentation mask, shape (batch_size, num_classes, H, W).
        target (torch.Tensor): The ground truth segmentation mask, shape (batch_size, H, W) or (batch_size, 1, H, W).

    Returns:
        float: The pixel accuracy as a percentage.
    """
    # Convert target to shape (batch_size, H, W) if it has a channel dimension
    if target.dim() == 4:
        target = target.squeeze(1)  # Shape (batch_size, H, W)
    
    # Get the predicted class with the highest probability
    pred = torch.argmax(pred, dim=1)  # Shape (batch_size, H, W)
    
    # Ensure target is in long format
    target = target.long()
    
    # Compute the number of correctly classified pixels
    correct_pixels = (pred == target).sum().item()
    
    # Compute the total number of pixels
    total_pixels = target.numel()
    
    # Calculate accuracy
    accuracy = correct_pixels / total_pixels
    
    return accuracy * 100  # Return as percentage

import torch

# Intersion over union validation calculation:



def evaluate_model(model, dataloader, device):
    print("evaluating the model")
    model.eval()
    total_accuracy = 0.0
    total_samples = 0
    iou_scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating metrics", leave=False):
            images = batch['image'].to(device)
            targets = batch['mask'].to(device)
            
            outputs = model(images)
            predicted = process_output(outputs)
            batch_accuracy = pixel_accuracy(outputs, targets)
            
            total_accuracy += batch_accuracy * images.size(0)
            total_samples += images.size(0)
            iou_scores.append(calculate_iou(predicted, targets.squeeze(1),21,[0,19])[0])

    mean_pixel_accuracy = total_accuracy / total_samples
    iou_score = np.average(iou_scores)
    print("total iou average = ",iou_score)
    return mean_pixel_accuracy


def calculate_iou(preds, labels, num_classes, ignore_index_array=None):
    iou_list = []
    gt_classes = set(torch.unique(labels).tolist())  

    with torch.no_grad():
        for cls in range(num_classes):
            if ignore_index_array is not None and cls in ignore_index_array:
                continue  # Skip the ignored class
            
            # Calculate intersection and union
            intersection = torch.sum((preds == cls) & (labels == cls)).item()
            union = torch.sum((preds == cls) | (labels == cls)).item()
            
            # Compute IoU for the class
            iou = intersection / union if union > 0 else 1
            iou_list.append((cls, iou))  # Store class ID with IoU value
            # print(f"Class {cls}: Intersection = {intersection}, Union = {union}, IoU = {iou}")

    # Filter IoU list to only include classes present in ground truth and not ignored
    valid_iou_list = [(cls, iou) for cls, iou in iou_list if cls in gt_classes]

    # Compute mean IoU, excluding classes that are not present in ground truth
    if len(valid_iou_list) > 0:
        mean_iou = sum(iou for _, iou in valid_iou_list) / len(valid_iou_list)
        print('mean iou = ', mean_iou)
    else:
        mean_iou = 0.0

    return mean_iou, iou_list

