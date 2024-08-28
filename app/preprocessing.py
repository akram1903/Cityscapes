
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from collections import defaultdict
from PIL import Image
from torchvision.transforms import Compose


class LoadImage():
    def __init__(self,keys):
        self.keys = keys
    def __call__(self,sample):
        
        for key in self.keys:
            sample[key] = Image.open(sample[key]).convert('RGB')
            #mask = Image.open(sample[key])
            
        return sample

class RotationAndFlipTransform:
    def __init__(self,keys ,rotation=15, flip=True):
        self.rotation = rotation
        self.flip = flip
        self.keys = keys

    def __call__(self, sample):
        # Apply random rotation
        if self.rotation:
            angle = T.RandomRotation.get_params([-self.rotation, self.rotation])
            for key in self.keys:
                sample[key] = F.rotate(sample[key], angle)
        
        # Apply random horizontal flip
        if self.flip and torch.rand(1) < 0.5:
            for key in self.keys:
                sample[key] = F.hflip(sample[key])
            
        return sample

class PixelValueTransform:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        """
        Apply pixel value transformations:
        - Replace 0 with 20
        - Replace 255 with 0
        """
        toTensor = T.ToTensor()
        pilToTensor = T.PILToTensor()
        toGrayscale = T.Grayscale(num_output_channels=1)  # Convert to grayscale
        toResize = T.Resize((256, 512),interpolation=T.InterpolationMode.NEAREST)
        
#         print("reached pixel transform")
        for key in self.keys:
            if key == "image":
                sample[key]= toTensor(sample[key])
            elif key == "mask":
                sample[key] = pilToTensor(sample[key])
            else:
                raise RuntimeError("key isn't image nor mask")
            sample[key]= toGrayscale(sample[key])
            sample[key]= toResize(sample[key])
        
        sample["mask"] = torch.where(sample["mask"] == 0, torch.tensor(20, dtype=torch.int), sample["mask"])
        sample["mask"] = torch.where(sample["mask"] > 20, torch.tensor(0, dtype=torch.int), sample["mask"])
        return sample


Train_data_transform=Compose([
   LoadImage( keys= ['image','mask'] ),
   RotationAndFlipTransform( keys= ['image','mask'] ),
    
   PixelValueTransform( keys= ['image','mask'] ),
    

])


Valid_data_transform=Compose([

   LoadImage( keys= ['image','mask'] ),
   PixelValueTransform( keys= ['image','mask'] ),    

])