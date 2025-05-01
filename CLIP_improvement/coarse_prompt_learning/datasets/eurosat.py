# from torchvision.datasets import EuroSAT

# from sklearn.model_selection import train_test_split
# from torch.utils.data import Subset

# def stratified_split(dataset, test_size=0.2):
#     labels = [dataset[i][1] for i in range(len(dataset))]  # Extract labels
#     train_indices, val_indices = train_test_split(
#         range(len(dataset)), test_size=test_size, stratify=labels, random_state=42
#     )
#     return Subset(dataset, train_indices), Subset(dataset, val_indices)

# def build_eurosat_dataset(data_path, transform):
#     dataset = EuroSAT(root=data_path, download=True, transform=transform)
    
#     # Create a 70/15/15 split for train/val/test
#     labels = [dataset[i][1] for i in range(len(dataset))]
    
#     # First split: 85% train+val, 15% test
#     train_val_indices, test_indices = train_test_split(
#         range(len(dataset)), test_size=0.15, stratify=labels, random_state=42
#     )
    
#     # Get labels for the train+val subset
#     train_val_labels = [labels[i] for i in train_val_indices]
    
#     # Second split: 82.35% train, 17.65% val (70/15 ratio of original dataset)
#     train_indices, val_indices = train_test_split(
#         train_val_indices, test_size=0.1765, stratify=train_val_labels, random_state=42
#     )
    
#     train_dataset = Subset(dataset, train_indices)
#     val_dataset = Subset(dataset, val_indices)
#     test_dataset = Subset(dataset, test_indices)
    
#     classnames = dataset.classes  # Get class names
#     fine_to_coarse = {
#         'AnnualCrop': 'Agricultural',
#         'Forest': 'Natural',
#         'HerbaceousVegetation': 'Natural',
#         'Highway': 'Other',
#         'Industrial': 'Urban',
#         'Pasture': 'Agricultural',
#         'PermanentCrop': 'Agricultural',
#         'Residential': 'Urban',
#         'River': 'Natural',
#         'SeaLake': 'Natural'
#     }  # Use the mapping from eurosat.py
#     coarse_labels = [fine_to_coarse[cls] for cls in classnames]
#     coarse_classes = sorted(set(fine_to_coarse.values()))  # Get unique coarse classes
#     coarse_to_index = {cls: idx for idx, cls in enumerate(coarse_classes)}
    
#     return train_dataset, val_dataset, test_dataset, classnames, coarse_labels, fine_to_coarse, coarse_to_index


import torch
from torchvision.datasets import  EuroSAT
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
import numpy as np

# Default transform for all datasets (matching CLIP requirements)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def build_eurosat_dataset(data_path, transform=transform, shots=0):
    # Load EuroSAT dataset
    full_dataset = EuroSAT(root=data_path, download=True, transform=transform)
    
    # Create an 80/20 train/val split, with test set separate
    labels = [full_dataset[i][1] for i in range(len(full_dataset))]
    train_val_indices, test_indices = train_test_split(
        range(len(full_dataset)), test_size=0.2, stratify=labels, random_state=42
    )
    train_val_labels = [labels[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=0.2, stratify=train_val_labels, random_state=42
    )
    
    # Few-shot sampling
    if shots > 0:
        train_indices = []
        labels = np.array(labels)
        classnames = full_dataset.classes
        for cls in range(len(classnames)):
            cls_indices = np.where(labels == cls)[0]
            selected_indices = np.random.choice(cls_indices, size=min(shots, len(cls_indices)), replace=False)
            train_indices.extend(selected_indices)
        train_indices = np.array(train_indices)
        np.random.shuffle(train_indices)
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Fine class names and coarse mapping
    classnames = full_dataset.classes
    fine_to_coarse = {
        'AnnualCrop': 'Agricultural',
        'Forest': 'Natural',
        'HerbaceousVegetation': 'Natural',
        'Highway': 'Other',
        'Industrial': 'Urban',
        'Pasture': 'Agricultural',
        'PermanentCrop': 'Agricultural',
        'Residential': 'Urban',
        'River': 'Natural',
        'SeaLake': 'Natural'
    }
    coarse_labels = [fine_to_coarse[cls] for cls in classnames]
    coarse_classes = sorted(set(fine_to_coarse.values()))
    coarse_to_index = {cls: idx for idx, cls in enumerate(coarse_classes)}
    
    return train_dataset, val_dataset, test_dataset, classnames, coarse_labels, fine_to_coarse, coarse_to_index