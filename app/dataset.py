from torch.utils.data import Dataset
import pandas as pd
from typing import List
from torchvision.transforms import Compose
from collections import defaultdict
from PIL import Image


class dataset(Dataset):
    def __init__(self, input_dataframe: pd.DataFrame, root_dir: str, KeysOfInterest: List[str], data_transform:Compose):
        self.root_dir = root_dir
        self.koi = KeysOfInterest
        self.input_dataframe = input_dataframe[self.koi]
        self.data_transform=data_transform

    def __getitem__(self, item: int):
        #image = Image.open(self.input_dataframe.iloc[item,0]).convert('RGB')
        #mask = Image.open(self.input_dataframe.iloc[item,1]).convert('L')
        sample={'image':self.input_dataframe.iloc[item,0],'mask':self.input_dataframe.iloc[item,1]}
        
        if self.data_transform:
            sample = self.data_transform(sample)
            
        return sample

    def __len__(self):
        return len(self.input_dataframe)