import torch

def process_output(output):
    """f
    Convert model output to class indices for each pixel.
    
    Args:
        output (torch.Tensor): The raw output from the model, shape (batch_size, num_classes, height, width).
    
    Returns:
        torch.Tensor: The processed output with shape (batch_size, height, width), where each pixel value is the class index.
    """
    # Apply argmax to get the class index with the highest score
    _, predicted = torch.max(output, 1)  # Shape (batch_size, height, width)
    return predicted

