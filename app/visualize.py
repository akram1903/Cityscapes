from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchmetrics

import matplotlib.patches as mpatches

from output_processing import process_output
from metrics import calculate_iou
# Example usage
def visualize_predictions(model, dataloader, device):
    """
    visualize only the first batch of the prediction
    """
    model.eval()
    print("visualizing the predictions")
    # Initialize IoU metric for multiclass segmentation
    iou_metric = torchmetrics.JaccardIndex(task='multiclass', num_classes=21).to(device)
    i = 0
    iou_score = 0
    # Extract class names and their corresponding colors from the labels
    trainid_to_name_color = {label[2]: (label[0], label[7]) for label in labels if label[2] != 255}

    # Separate the names and colors into two lists
    class_names = [name for trainId, (name, color) in trainid_to_name_color.items()]
    colors = [color for trainId, (name, color) in trainid_to_name_color.items()]
    patches = [mpatches.Patch(color=np.array(color)/255.0, label=name) for name, color in zip(class_names, colors)]
    
    # plt.figure(figsize=(2, 2))
    # plt.legend(handles=patches, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=2)
    # plt.axis('off')  # Turn off the axis
    # plt.title('Class Labels and Colors')
    # plt.show()
    Path('my_directory').mkdir(exist_ok=True)
    
    with torch.no_grad():
        for batch in dataloader:
            i += 1
            images = batch['image'].to(device)

            if(len(images.shape)<4):
                images = images.unsqueeze(0)  # Adds a batch dimension if missing
            # print("len shape: ",len(images.shape))
            targets = batch['mask'].to(device)
            
            outputs = model(images)
            predicted = process_output(outputs)  # Shape (batch_size, height, width)
            
            # Calculate IoU for this batch
            
            # Convert to numpy array for visualization
            predicted_np = predicted.cpu().numpy()
            
            for i in range(predicted_np.shape[0]):  # Iterate over batch
                fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        
                predicted_color = map_trainid_to_color(predicted_np[i], trainid_to_color)

                ax[0].imshow(predicted_color)  # Use a colormap suitable for your classes
                ax[0].set_title('Predicted Segmentation')
                
                mask_color = map_trainid_to_color(targets[i].cpu().squeeze().numpy(),trainid_to_color)
                # If you want to show the ground truth mask as well:
                ax[1].imshow(mask_color)
                ax[1].set_title('Ground Truth Mask')
                # plt.show()
                plt.savefig(f'./outputs/plot_{i}.png')
            break  
        iou_score = calculate_iou(predicted, targets.squeeze(1),21,[0,19])[0]
        print("iou_score for that batch: ",iou_score)
            


def map_trainid_to_color(mask, trainid_to_color):
    # Assuming mask is a 2D numpy array with shape (height, width)
    height, width = mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for trainId, color in trainid_to_color.items():
        color_mask[mask == trainId] = color
    
    return color_mask



# Define the labels as a list of tuples, as provided in the comment.
labels = [
    ('unlabeled', 0, 0, 'void', 0, False, True, (0, 0, 0)),
    ('ego vehicle', 1, 0, 'void', 0, False, True, (0, 0, 0)),
    ('rectification border', 2, 0, 'void', 0, False, True, (0, 0, 0)),
    ('out of roi', 3, 0, 'void', 0, False, True, (0, 0, 0)),
    ('static', 4, 0, 'void', 0, False, True, (0, 0, 0)),
    ('dynamic', 5, 0, 'void', 0, False, True, (111, 74, 0)),
    ('ground', 6, 0, 'void', 0, False, True, (81, 0, 81)),
    ('road', 7, 20, 'ground', 1, False, False, (128, 64, 128)),
    ('sidewalk', 8, 1, 'ground', 1, False, False, (244, 35, 232)),
    ('parking', 9, 0, 'ground', 1, False, True, (250, 170, 160)),
    ('rail track', 10, 0, 'ground', 1, False, True, (230, 150, 140)),
    ('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
    ('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
    ('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
    ('guard rail', 14, 0, 'construction', 2, False, True, (180, 165, 180)),
    ('bridge', 15, 0, 'construction', 2, False, True, (150, 100, 100)),
    ('tunnel', 16, 0, 'construction', 2, False, True, (150, 120, 90)),
    ('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
    ('polegroup', 18, 0, 'object', 3, False, True, (153, 153, 153)),
    ('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
    ('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
    ('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
    ('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
    ('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
    ('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
    ('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
    ('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    ('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
    ('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    ('caravan', 29, 0, 'vehicle', 7, True, True, (0, 0, 90)),
    ('trailer', 30, 0, 'vehicle', 7, True, True, (0, 0, 110)),
    ('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    ('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    ('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    ('license plate', 34, 19, 'vehicle', 7, False, True, (0, 0, 142)),
]

# Create a mapping from trainId to color
trainid_to_color = {label[2]: label[7] for label in labels if label[2] != 255}

