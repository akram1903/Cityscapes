import torch
import torch.nn as nn
import torch.nn.functional as Torch_F
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


import preprocessing
import dataset
from metrics import evaluate_model
from csv_create import create_image_mask_csv

num_classes = 21
# IMAGE_ROOT_DIRECTORY = '/home/akram/Downloads/archive/kaggle/input/cityscapes/Cityspaces/images/train/'
# MASK_ROOT_DIRECTORY = '/home/akram/Downloads/archive/kaggle/input/cityscapes/Cityspaces/gtFine/train/'

IMAGE_ROOT_DIRECTORY = '/app/data/images/train'
MASK_ROOT_DIRECTORY = '/app/data/gtFine/train'
IMAGE_MASK_MAPPING_PATH = './cityscapes_image_mask_mapping.csv'
TRAIN_CSV = './cityscapes_train.csv'
TEST_CSV = './cityscapes_test.csv'
VAL_CSV = './cityscapes_cv.csv'


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc_conv1 = self.double_conv(in_channels, 64)
        self.enc_conv2 = self.double_conv(64, 128)
        self.enc_conv3 = self.double_conv(128, 256)
        self.enc_conv4 = self.double_conv(256, 512)
        
        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv4 = self.double_conv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = self.double_conv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.double_conv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.double_conv(128, 64)
        
        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(Torch_F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc_conv3(Torch_F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc_conv4(Torch_F.max_pool2d(enc3, kernel_size=2))
        
        # Bottleneck
        bottleneck = self.bottleneck(Torch_F.max_pool2d(enc4, kernel_size=2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec_conv4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec_conv3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec_conv2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec_conv1(dec1)
        
        return self.out_conv(dec1)

if __name__ == "__main__":

    
    output_csv_file = './cityscapes_image_mask_mapping.csv'

    create_image_mask_csv(IMAGE_ROOT_DIRECTORY, MASK_ROOT_DIRECTORY, output_csv_file)


    # Load the data from the CSV file
    df = pd.read_csv(IMAGE_MASK_MAPPING_PATH)

    # Split the data into train and test sets
    _, test_df = train_test_split(df, test_size=0.1, random_state=42)
    test_df.to_csv(TEST_CSV, index=False)

    # print(f"Training data saved to 'cityscapes_train.csv' with {len(train_df)} samples.")
    print(f"Testing data saved to {TEST_CSV} with {len(test_df)} samples.")


    # Instantiate the model
    model = UNet(in_channels=1, out_channels=21)

    # Initilize your dataset

    ds_test=dataset.dataset(input_dataframe=pd.read_csv(TEST_CSV),
                    root_dir="",
                    KeysOfInterest=["image","mask"],
                    data_transform=preprocessing.Valid_data_transform)
    
    dl_test = DataLoader(dataset=ds_test,batch_size= 4 ,num_workers=1 ,prefetch_factor=8,shuffle=True)
    # load the model 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 2: Load the saved state dictionary
    model.load_state_dict(torch.load('./DiceMoreTrain4_CPU.pth',map_location=device,weights_only=True))

    # Step 3: Set the model to evaluation mode (if needed)
    model.eval()


    # visualize_predictions(model, dl_test, device)
    evaluate_model(model,dl_test,device)