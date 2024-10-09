import torch
from torchvision import transforms 
import os
import pandas as pd
from dataloader.SkinCancerDataset import SkinCancerDataset
import config
from concurrent.futures import ThreadPoolExecutor
import os
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from model.resnet import ResidualBlock, ResNet
from PIL import Image 
import matplotlib.pyplot as plt

# File paths
data_dir = '/kaggle/input/isic-2024-challenge/'
image_dir = os.path.join(data_dir, 'train-image/image')

# Load metadata
metadata_path = os.path.join(data_dir, 'train-metadata.csv')
metadata_df = pd.read_csv(metadata_path)

positive_class_df = metadata_df[metadata_df['target'] == 1]
negative_class_df = metadata_df[metadata_df['target'] == 0]

num_positive_cases = positive_class_df.shape[0]

def load_images_in_parallel(metadata_df, image_dir, num_workers=4):
    def load_image(row):
        isic_id = row["isic_id"]
        label = row["target"]
        image_path = os.path.join(image_dir, f"{isic_id}.jpg")
        if os.path.exists(image_path):
            img = Image.open(image_path)
            return img, label
        else:
            print(f"Image {image_path} not found!")
            return None, label
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(load_image, row) for _, row in metadata_df.iterrows()]
        images = []
        labels = []
        for future in futures:
            img, label = future.result()
            if img is not None:
                images.append(img)
                labels.append(label)
        return images, labels

# configurable parameters
total_images = 1000
num_workers = 8 
positive_images = positive_class_df
# calculate the number of negative images to load to maintain the total image count
num_negative_images = total_images - num_positive_cases
    
#randomly sample negative images to fill the remaining number
negative_images_sampled = negative_class_df.sample(num_negative_images, random_state=28)
    
# concatenate positice and sampled negative images
final_metadata_df = pd.concat([positive_images, negative_images_sampled])
    
loaded_images, labels = load_images_in_parallel(final_metadata_df, image_dir, num_workers=num_workers)

loaded_positive_images = [img for img in zip(loaded_images, labels) if img[1] == 1]
loaded_negative_images = [img for img in zip(loaded_images, labels) if img[1] == 0]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])
positive_labels = [1] * len(loaded_positive_images)
negative_labels = [0] * len(loaded_negative_images)

positive_dataset = SkinCancerDataset(loaded_positive_images, positive_labels, transform=transform)
negative_dataset = SkinCancerDataset(loaded_negative_images, negative_labels, transform=transform)

full_images = loaded_positive_images + loaded_negative_images
full_labels = positive_labels + negative_labels

full_dataset = SkinCancerDataset(full_images, full_labels, transform=transform)
dataloader = DataLoader(full_dataset, batch_size=32, shuffle=True, worker_init_fn=torch.utils.data.get_worker_info())

model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=config.num_classes)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 1)
model = torch.nn.Sequential(model, torch.nn.Sigmoid())
model.to(config.device)

# Define the loss function
criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

loss_values = []
for epoch in range(config.num_epochs):
    running_loss = 0.0  
    model.train()
    for images, labels in dataloader:
        images = images.to(config.device)
        labels = labels.float().to(config.device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = torch.nn.BCELoss()(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(dataloader)
    loss_values.append(epoch_loss)
    print(f"Epoch [{epoch + 1} / {config.num_epochs}], Loss: {epoch_loss: .4f}")
    
# Plotting training loss
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()