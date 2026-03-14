
import sys
import os
import torchvision.transforms as transforms
import pydicom as pyd
import numpy as np
from PIL import Image
import wfdb
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from numba import jit
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import public
from tensorflow.python.platform.benchmark import OVERRIDE_GLOBAL_THREADPOOL
from torch.utils.data import Dataset, DataLoader
import opendatasets as od
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#steps: list based approach by saving items to lists when there's inaccessible for loops

# Define download path
path_download = "salikhussaini49/sunnybrook-cardiac-mri" #images
dataset_path = kagglehub.datasets.dataset_download(path_download)

img_files = [] #store files
processed_image_files = [] #store processed files

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigderivative(x):
    return x * (1 - x)


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def modify_list(input_list):

    mid_index = len(input_list) // 2  # Find the midpoint of the list
    modified_list = []
    modified_list.extend([1] * mid_index)  # Add 1s for the first half
    modified_list.extend([0] * (len(input_list) - mid_index))  # Add 0s for the second half
    return modified_list


def imageProcess(imgpath): #only for this project dosent apply to all
    img = pyd.dcmread(imgpath)
    img = img.pixel_array
    img = Image.fromarray(img)
    img = img.convert('RGB')
    img = img.resize((256,256))

    img = np.array(img) #yes
    img = (img - 127.5) / 127.5 # -1 to 1
    return img

# Explore the directory structure
def explore_img_directory(path, level=0):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if not os.path.isdir(item_path):
            img_files.append(item_path)
        if os.path.isdir(item_path):
            explore_img_directory(item_path, level + 1)
explore_img_directory(dataset_path)

counter = 1
for img in img_files:
    if img.endswith('.dcm'):
        processx = imageProcess(img)
        processed_image_files.append(processx)
        counter += 1
    else:
        print("Not processed")

label_files = modify_list([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]) #now its the same length so no size errors

# Verify the CSV file path
#print("Full path to CSV file:", full_path)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)



# Load PTB-XL metadata
# may use later
#df = pd.read_csv(full_path, index_col=0)
#if 'ecg_id' in df.columns:
#    df.set_index('ecg_id', inplace=True)
#df['scp_codes'] = df['scp_codes'].apply(eval)  # Evaluate string representations of dictionaries

# Define CAD-related diagnostic classes
#cad_classes = ['ASMI', 'AMI', 'LMI', 'IMI']
# Create binary labels for CAD
#df['cad'] = df['scp_codes'].apply(lambda x: any(key in cad_classes for key in x.keys()))

# Function to load ECG data
def load_ecg(filename):
    sig, _ = wfdb.rdsamp(filename)
    return sig[:, 1]  # Return only lead II

class ProcessedImageDataset(Dataset):
    def __init__(self, processed_img_files, labels, transform=None):
        self.processed_img_files = processed_img_files
        self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.processed_img_files)

    def __getitem__(self, index):
        image = self.processed_img_files[index]
        label = self.label[index]
        label = self.label[index]
        label = torch.tensor(label, dtype=torch.float32)

        #if isinstance(image, np.ndarray):
         #   image = torch.tensor(image, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

# Define the Temporal Attention Layer
class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super(TemporalAttention, self).__init__()

    def forward(self, x, chunk_size = 1024):
        batch_size, _, total_length = x.shape
        #output = []

        weights = torch.softmax(x, dim=-1) # max_dim as length, width, and rgb
        return weights * x

# Define the CNN Model with Temporal Attention
class CADDetector(nn.Module):
    def __init__(self):
        super(CADDetector, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.attention = TemporalAttention(131072)
        self.fc1 = nn.Linear(128*32*32, 256) #matches size
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = x.float()
        #print(x.shape)
        #Line below may be wrong
        x = x.view(x.size(0), 3, 256, 256) # To batch size, channels, height and width
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # flattens shape
        x = x.unsqueeze(1)
        x = self.attention(x).squeeze(1)
        x = torch.relu(self.fc1(x))
        finished = torch.sigmoid(self.fc(x).squeeze(1)).to(torch.float)
        #print(finished)
        return finished.float()


def BCE_loss_func(pred,true):
    pred = torch.clamp(pred, min=1e-7, max=1-1e-7)

    loss = -(true * torch.log(pred) + (1- true) * torch.log(1-pred))
    return loss.mean()

def bin_cross_entropy_error(pred,true):

    loss_BCE = BCE_loss_func(pred,true)
    #print(f"{loss_BCE} line 185")

    return loss_BCE

def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0

    for batch in dataloader:
        inputs, labels = batch
        optimizer.zero_grad()

        predictions = model(inputs)

        loss = bin_cross_entropy_error(predictions, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Prepare Data and Good news no errors
train_df, test_df,train_labels,test_labels = train_test_split(processed_image_files, label_files, test_size=0.1, random_state=42)
train_dataset = ProcessedImageDataset(processed_img_files=train_df, labels=train_labels, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset =  ProcessedImageDataset(processed_img_files=test_df, labels=test_labels, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Initialize Model, Loss Function, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CADDetector().to(device).float()
optimizer = optim.Adam(model.parameters(), lr=0.101)

# Training Loop
epoch_time = 0
num_epochs = 100


# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for ecgs, labels in test_loader:
        ecgs, labels = ecgs.to(device), labels.to(device)
        outputs = model(ecgs)
        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()
print(all_preds.item(5))
# Calculate Metrics
accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
precision = precision_score(all_labels, (all_preds > 0.5).astype(int),average="micro")
recall = recall_score(all_labels, (all_preds > 0.5).astype(int),average="micro")
#auc = roc_auc_score(all_labels, all_preds,multi_class='ovo')
for epoch in range(num_epochs):
    train(model,train_loader,optimizer)
    epoch_time +=1
    print(f"On epoch {epoch_time}")
    print(accuracy)
    print(precision)
    print(recall)
# save to file


print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

#print(f'AUC: {auc:.4f}')
 #resets

# Save the Model
torch.save(model.state_dict(), 'cad_detector_model.pth')