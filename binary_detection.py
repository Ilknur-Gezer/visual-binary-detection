import os
import kornia
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
from tqdm.notebook import tqdm
from torchvision import models
from typing import Tuple, List
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# FLAGS
# Change the following to {True, False} to enable or disable the execution of generating .csv dataset.
RECREATE_DATASET = True
TRAINING = True
TESTING = True

# Data
# Generate CSV files
# The following generates .csv files for training, validation and testing, based on .fits files placed in binary and else
# folders, and labels them 1 and 0, respectively.

def create_combined_csv(root_folder: str, output_folder: str, output_file: str = 'combined.csv'):
    """
    Creates a combined CSV file from .fits files in the given folders.
    
    Parameters:
    root_folder (str): The root folder containing the 'binary' and 'else' folders.
    output_folder (str): The folder where the CSV files will be saved.
    output_file (str): The name of the output combined CSV file.
    """
    # Define the subfolders and corresponding labels
    folders = {'binary': 1, 'else': 0}
    data = []

    # Iterate over the folders and files
    for folder, label in folders.items():
        folder_path = os.path.join(root_folder, folder)
        for file in os.listdir(folder_path):
            # Consider only FITS files
            if file.endswith('.fits'):
                # Save the absolute path, whith the corresponding label
                data.append({'path': os.path.abspath(os.path.join(folder_path, file)), 'label': label})

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(data)
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(os.path.join(output_folder, output_file), index=False)


def stratified_split(input_df: pd.DataFrame,
                     train_size: int = 0.70,
                     test_size: int = 0.15,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs a stratified split on the input DataFrame.
    
    Parameters:
    input_df (DataFrame): The input DataFrame to be split.
    train_size (float): The proportion of the dataset to include in the train split.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.
    
    Returns:
    tuple: A tuple containing the train, validation, and test DataFrames.
    """
    # Split the data into training and temporary (validation + test)
    train_df, temp_df = train_test_split(input_df, train_size=train_size, stratify=input_df['label'],
                                         random_state=random_state)

    # Split the temp data into validation and test
    val_size = test_size / (1 - train_size)
    val_df, test_df = train_test_split(temp_df, test_size=val_size, stratify=temp_df['label'],
                                       random_state=random_state)

    return train_df, val_df, test_df


def save_csv_files(dfs: List[pd.DataFrame],
                   output_folder: str,
                   filenames: List[str] = ['train.csv', 'valid.csv', 'test.csv']):
    """
    Saves given DataFrames as CSV files in the specified folder.
    
    Parameters:
    dfs (list of DataFrame): List of DataFrames to be saved.
    output_folder (str): The folder where the CSV files will be saved.
    filenames (list of str): Names of the output CSV files.
    """
    for df, filename in zip(dfs, filenames):
        df.to_csv(os.path.join(output_folder, filename), index=False)


# Run the following lines only if there has been a change in the dataset, so you want to recreate the .csv files. 
# Once you have created the .csv files, you can set the RECREATE_DATASET flag to False so that it does not do it again.
root_folder = 'dataset'
output_folder = os.path.join('dataset', 'csv')

if RECREATE_DATASET:
    # Create combined CSV
    create_combined_csv(root_folder, output_folder)

    # Load the combined CSV
    combined_df = pd.read_csv(os.path.join(output_folder, 'combined.csv'))

    # Perform stratified split
    train_df, val_df, test_df = stratified_split(combined_df)

    # Save the split datasets
    save_csv_files([train_df, val_df, test_df], output_folder)


# Dataset and Data Augmentations
# Define Custom Dataset

class FitsDataset(Dataset):
    """
    A custom PyTorch Dataset for handling .fits files.

    Args:
        csv_file (str): Path to the CSV file containing paths and labels.
        transform (callable, optional): Optional transform (data augmentation) to be applied on a sample.

    Attributes:
        data_frame (DataFrame): Pandas DataFrame containing the file paths and labels.
        transform (callable): Transform to be applied on a sample.
    """

    def __init__(self, csv_file: str, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        fits_path = self.data_frame.iloc[idx, 0]  # first column of .csv file
        label = self.data_frame.iloc[idx, 1]  # second column of .csv file

        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data

        # Normalize the image data to [0,1]
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        image_data = (image_data - min_val) / (max_val - min_val)
        image_data = image_data.astype(np.float32)
        
        # Create Tensor
        image_data = torch.from_numpy(image_data)
        # add channel dimension
        image_data = image_data.unsqueeze(0)
        
        # Apply data augmentation if defined
        if self.transform:
            image_data = self.transform(image_data)

        # Center crop to size
        image_data = transforms.functional.center_crop(image_data, output_size=(224, 224))
        
        return image_data, label

# Define Data Augmentation Pipeline     

class CustomTransform:
    def __init__(self, p=0.5):
        self.p = p
        self.angles = [0, 90, 180, 270]
        self.translation_values = [-1, 0, 1]

    def __call__(self, input_data):

        # Random horizontal flip with probability p
        if random.random() < self.p:
            input_data = torch.flip(input_data, [-1])

        # Random vertical flip with probability p
        if random.random() < self.p:
            input_data = torch.flip(input_data, [-2])

        # Random translation with one of the values
        trans_x = random.choice(self.translation_values)
        trans_y = random.choice(self.translation_values)
        translation_value = torch.tensor([[trans_x, trans_y]], dtype=torch.float32)
        input_data = kornia.geometry.transform.translate(input_data, translation_value, mode='nearest')

        # Random rotation with one of the angles
        angle = random.choice(self.angles)
        angle = torch.tensor(angle, dtype=torch.float32)
        # # Add batch dimension
        # input_data = input_data.unsqueeze(0)
        input_data = kornia.geometry.transform.rotate(input_data, angle, mode='bicubic')

        return input_data

# Training
# Setup
if TRAINING:

    # Load ResNet model without pre-trained weights
    model = models.resnet34(pretrained=False)
    # Change input channels from 3 to 1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Add additional layers
    num_features = model.fc.in_features
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Replace the average pooling layer with adaptive average pooling
    model.layer_extra = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )
    # Change output channels to 2 (binary classification)
    model.fc = nn.Linear(model.fc.in_features, 2)
    #model.fc = nn.Linear(512, 2)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.05)

    # Save the best model to this path
    best_model_path = os.path.join("trained_models")
    os.makedirs(best_model_path, exist_ok=True)
    best_model = os.path.join(best_model_path, f'best_model.pth')

    # Check if GPU is available and move the model to GPU if it is
    # Specify the GPU to use (e.g., GPU 0)
    target_gpu = 1
    device = torch.device(f"cuda:{target_gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {torch.cuda.get_device_name(target_gpu)}")

    #device = torch.device(gip if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    num_epochs = 40  # Number of training epochs

# Training and Validation Loops
 # training the model
    best_f1 = 0.0
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        model.train()
        train_preds, train_targets = [], []
        train_losses = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            train_losses.append(loss.item())

        train_f1 = f1_score(train_targets, train_preds, average='weighted')

        model.eval()
        valid_preds, valid_targets = [], []
        valid_losses = []

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                preds = outputs.argmax(dim=1)
                valid_preds.extend(preds.cpu().numpy())
                valid_targets.extend(labels.cpu().numpy())

                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())

        valid_f1 = f1_score(valid_targets, valid_preds, average='weighted')

        if valid_f1 >= best_f1:
            best_f1 = valid_f1
            torch.save(model.state_dict(), best_model)

        scheduler.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Train F1: {train_f1:.4f}, Valid F1: {valid_f1:.4f}')

# Let's test our trained model!
# Define test dataset
if TESTING:
    
    test_csv = os.path.join('dataset', 'csv', 'test.csv')
    test_dataset = FitsDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load the best trained model
# Let's see how it works on unseen data   

    test_preds, test_targets = [], []
    test_losses = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

    # Generate a classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_targets, test_preds))

# Define Production Dataset
class ProductionFitsDataset(Dataset):
    
    """
    A custom PyTorch Dataset for handling .fits files.

    Args:
        csv_file (str): Path to the CSV file containing paths and labels.

    Attributes:
        data_frame (DataFrame): Pandas DataFrame containing the file paths and labels.
    """

    def __init__(self, csv_file: str):
        self.data_frame = pd.read_csv(csv_file)
   
    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        fits_path = self.data_frame.iloc[idx, 0]

        with fits.open(fits_path) as hdul:
            image_data = hdul[0].data

        # Normalize the image data to [0,1]
        min_val = np.min(image_data)
        max_val = np.max(image_data)
        image_data = (image_data - min_val) / (max_val - min_val)
        image_data = image_data.astype(np.float32)

        # Create Tensor
        image_data = torch.from_numpy(image_data)
        # add channel dimension
        image_data = image_data.unsqueeze(0)

        # Center crop to size
        image_data = transforms.functional.center_crop(image_data, output_size=(224, 224))

        return image_data

# Create Production CSV
def create_production_csv(root_folder: str, output_folder: str, output_file: str = 'production.csv'):
    """
    Creates a combined CSV file from .fits files in the given folder.

    Parameters:
    root_folder (str): The root folder containing the FITS files.
    output_folder (str): The folder where the CSV file will be saved.
    output_file (str): The name of the output CSV file.
    """

    
    root_folder = '/dataplus3/ilknur/stamp_images/binary_stamps/panstarr/panstarr_stacked/fits'
    output_folder = '/home/jupyter-ilknur/BinaryDetection/panstarr/production'
    
    prod_data = []

    # Iterate over the files in the root folder
    for file in os.listdir(root_folder):
        # Consider only FITS files
        if file.endswith('.fits'):
            # Save the absolute path
            prod_data.append({'path': os.path.abspath(os.path.join(root_folder, file))})
            
    # Create a DataFrame and save as CSV
    os.makedirs(output_folder, exist_ok=True)
    prod_df = pd.DataFrame(prod_data)
    prod_df.to_csv(os.path.join(output_folder, output_file), index=False)
    
create_production_csv(root_folder, output_folder)

# Load the Production Dataset and Make Predictions
# Load dataset
prod_csv = ('/home/jupyter-ilknur/BinaryDetection/panstarr/production/production.csv')
prod_dataset = ProductionFitsDataset(prod_csv)
prod_loader = DataLoader(prod_dataset, batch_size=16, shuffle=False)
    
    
    
# Define and load your model
best_model_path = os.path.join('/home/jupyter-ilknur/BinaryDetection/panstarr/trained_models')
best_model = os.path.join(best_model_path, f'best_model.pth')
# Load the best model
model.load_state_dict(torch.load(best_model))
model.eval()
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Make predictions
    
predictions = []

with torch.no_grad():
    for images in prod_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        predictions.extend(preds.cpu().numpy())
            

# Now 'predictions' contains the predicted labels for your production dataset
#print(predictions)

# Save predictions to CSV
pred_df = pd.DataFrame({'Predictions': predictions})
pred_df.to_csv('/home/jupyter-ilknur/BinaryDetection/panstarr/production/predictions.csv', index=False)

# Combine the Production Dataset with the Predictions¶
T1 = pd.read_csv('/home/jupyter-ilknur/BinaryDetection/panstarr/production/production.csv')
df1 = pd.DataFrame(T1)
T2 = pd.read_csv('/home/jupyter-ilknur/BinaryDetection/panstarr/production/predictions.csv')
df2 = pd.DataFrame(T2)
result = pd.concat([df1, df2], axis=1)
result.to_csv('/home/jupyter-ilknur/BinaryDetection/panstarr/results.csv', index=False)