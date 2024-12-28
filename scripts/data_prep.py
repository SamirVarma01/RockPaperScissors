import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data Preprocessing, transforms data to account for randomness and variety in how people present rock, paper, scissors
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

#Loads datasets
train_data = datasets.ImageFolder('/dataset/train', transform=train_transforms)
val_data = datasets.ImageFolder('/dataset/val', transform=val_test_transforms)
test_data = datasets.ImageFolder('/dataset/test', transform=val_test_transforms)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print("Data loaded successfully!")
