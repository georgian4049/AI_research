import torch
from torchvision.datasets import CIFAR100
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def build_cifar100_dataset(data_path, transform=transform, shots=0):
    # Load CIFAR-100 datasets
    train_full_dataset = CIFAR100(root=data_path, train=True, download=True, transform=transform)
    test_dataset = CIFAR100(root=data_path, train=False, download=True, transform=transform)
    
    # Fine class names and coarse mapping
    classnames = train_full_dataset.classes
    fine_to_coarse = {
        'apple': 'fruit_and_vegetables', 'aquarium_fish': 'fish', 'baby': 'people', 'bear': 'large_carnivores',
        'beaver': 'medium_mammals', 'bed': 'household_furniture', 'bee': 'insects', 'beetle': 'insects',
        'bicycle': 'vehicles_1', 'bottle': 'household_items', 'bowl': 'household_items', 'boy': 'people',
        'bridge': 'large_man-made_outdoor_things', 'bus': 'vehicles_1', 'butterfly': 'insects',
        'camel': 'large_omnivores_and_herbivores', 'can': 'household_items', 'castle': 'large_man-made_outdoor_things',
        'caterpillar': 'insects', 'cattle': 'large_omnivores_and_herbivores', 'chair': 'household_furniture',
        'chimpanzee': 'large_omnivores_and_herbivores', 'clock': 'household_items', 'cloud': 'large_natural_outdoor_scenes',
        'cockroach': 'insects', 'couch': 'household_furniture', 'crab': 'aquatic_mammals', 'crocodile': 'reptiles',
        'cup': 'household_items', 'dinosaur': 'reptiles', 'dolphin': 'aquatic_mammals', 'elephant': 'large_omnivores_and_herbivores',
        'flatfish': 'fish', 'forest': 'large_natural_outdoor_scenes', 'fox': 'medium_mammals', 'girl': 'people',
        'hamster': 'small_mammals', 'house': 'large_man-made_outdoor_things', 'kangaroo': 'large_omnivores_and_herbivores',
        'keyboard': 'household_items', 'lamp': 'household_furniture', 'lawn_mower': 'vehicles_2', 'leopard': 'large_carnivores',
        'lion': 'large_carnivores', 'lizard': 'reptiles', 'lobster': 'aquatic_mammals', 'man': 'people',
        'maple_tree': 'trees', 'motorcycle': 'vehicles_1', 'mountain': 'large_natural_outdoor_scenes',
        'mouse': 'small_mammals', 'mushroom': 'fruit_and_vegetables', 'oak_tree': 'trees', 'orange': 'fruit_and_vegetables',
        'orchid': 'flowers', 'otter': 'aquatic_mammals', 'palm_tree': 'trees', 'pear': 'fruit_and_vegetables',
        'pickup_truck': 'vehicles_1', 'pine_tree': 'trees', 'plain': 'large_natural_outdoor_scenes',
        'plate': 'household_items', 'poppy': 'flowers', 'porcupine': 'medium_mammals', 'possum': 'medium_mammals',
        'rabbit': 'small_mammals', 'raccoon': 'medium_mammals', 'ray': 'fish', 'road': 'large_man-made_outdoor_things',
        'rocket': 'vehicles_2', 'rose': 'flowers', 'sea': 'large_natural_outdoor_scenes', 'seal': 'aquatic_mammals',
        'shark': 'fish', 'shrew': 'small_mammals', 'skunk': 'medium_mammals', 'skyscraper': 'large_man-made_outdoor_things',
        'snail': 'insects', 'snake': 'reptiles', 'spider': 'insects', 'squirrel': 'small_mammals',
        'streetcar': 'vehicles_1', 'sunflower': 'flowers', 'sweet_pepper': 'fruit_and_vegetables',
        'table': 'household_furniture', 'tank': 'vehicles_2', 'telephone': 'household_items',
        'television': 'household_furniture', 'tiger': 'large_carnivores', 'tractor': 'vehicles_2',
        'train': 'vehicles_1', 'trout': 'fish', 'tulip': 'flowers', 'turtle': 'reptiles',
        'wardrobe': 'household_furniture', 'whale': 'aquatic_mammals', 'willow_tree': 'trees',
        'wolf': 'large_carnivores', 'woman': 'people', 'worm': 'insects'
    }
    coarse_labels = [fine_to_coarse[cls] for cls in classnames]
    coarse_classes = sorted(set(fine_to_coarse.values()))
    coarse_to_index = {cls: idx for idx, cls in enumerate(coarse_classes)}
    
    # Split training data into train and validation
    labels = train_full_dataset.targets
    train_indices, val_indices = train_test_split(
        range(len(train_full_dataset)), test_size=0.2, stratify=labels, random_state=42
    )
    
    if shots > 0:
        # Create a 16-shot training dataset
        train_indices = []
        labels = np.array(train_full_dataset.targets)
        for cls in range(len(classnames)):
            cls_indices = np.where(labels == cls)[0]
            selected_indices = np.random.choice(cls_indices, size=min(shots, len(cls_indices)), replace=False)
            train_indices.extend(selected_indices)
        train_indices = np.array(train_indices)
        np.random.shuffle(train_indices)
    
    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(train_full_dataset, val_indices)
    
    return train_dataset, val_dataset, test_dataset, classnames, coarse_labels, fine_to_coarse, coarse_to_index

