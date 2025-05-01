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

def build_cifar100_small_dataset(data_path, transform=transform, train_samples_per_class=50, val_samples_per_class=10):
    # Load CIFAR-100 (50,000 train images, 100 fine classes, 20 coarse classes)
    full_dataset = CIFAR100(root=data_path, train=True, download=True, transform=transform)
    
    # Get labels and indices
    labels = np.array(full_dataset.targets)
    num_classes = len(full_dataset.classes)  # 100
    
    # Ensure requested samples don't exceed available data (500 per class)
    train_samples_per_class = min(train_samples_per_class, 500)
    val_samples_per_class = min(val_samples_per_class, 500 - train_samples_per_class)
    
    train_indices = []
    val_indices = []
    
    # Stratified sampling: select fixed samples per class
    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0]
        np.random.shuffle(class_indices)  # Randomize order
        train_indices.extend(class_indices[:train_samples_per_class])
        val_indices.extend(class_indices[train_samples_per_class:train_samples_per_class + val_samples_per_class])
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Fine class names
    classnames = full_dataset.classes
    
    # Official coarse-to-fine mapping (unchanged)
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
    
    print(f"Train dataset size: {len(train_dataset)} images, Val dataset size: {len(val_dataset)} images")
    return train_dataset, val_dataset, classnames, coarse_labels, fine_to_coarse, coarse_to_index