from torchvision.datasets import DTD
from torchvision import transforms

def build_dtd_dataset(data_path, transform):
    _transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30), # Higher rotation for texture variations
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # DTD doesn't follow ImageNet normalization
    ])
    
    # For test set, we don't need data augmentation
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = DTD(root=data_path, split='train', download=True, transform=_transform)
    val_dataset = DTD(root=data_path, split='val', download=True, transform=_transform)
    test_dataset = DTD(root=data_path, split='test', download=True, transform=test_transform)
    
    classnames = train_dataset.classes
    # Define fine-to-coarse mapping manually
    fine_to_coarse = {
    'banded': 'Geometric',
    'blotchy': 'Organic',
    'braided': 'Textile',
    'bubbly': 'Organic',
    'bumpy': 'Rough',
    'chequered': 'Geometric',  # Corrected capitalization
    'cobwebbed': 'Organic',
    'cracked': 'Rough',
    'crosshatched': 'Geometric',
    'crystalline': 'Mineral',
    'dotted': 'Geometric',  # Added missing class
    'fibrous': 'Textile',
    'flecked': 'Organic',
    'freckled': 'Organic',
    'frilly': 'Textile',
    'gauzy': 'Textile',
    'grid': 'Geometric',
    'grooved': 'Rough',
    'honeycombed': 'Geometric',
    'interlaced': 'Textile',
    'knitted': 'Textile',  # Corrected spelling
    'lacelike': 'Textile',
    'lined': 'Geometric',
    'marbled': 'Mineral',
    'matted': 'Textile',
    'meshed': 'Geometric',
    'paisley': 'Organic',  # Added missing class
    'perforated': 'Artificial',
    'pitted': 'Rough',
    'pleated': 'Textile',
    'polka-dotted': 'Geometric',
    'porous': 'Mineral',
    'pothole': 'Rough',  # Added missing class
    'scaly': 'Organic',
    'smeared': 'Organic',
    'spiralled': 'Geometric',
    'sprinkled': 'Organic',
    'stained': 'Organic',
    'stratified': 'Mineral',
    'striped': 'Geometric',
    'studded': 'Artificial',
    'swirly': 'Geometric',
    'veined': 'Mineral',
    'waffled': 'Textile',
    'woven': 'Textile',
    'wrinkled': 'Organic',
    'zigzagged': 'Geometric'
}
    coarse_labels = [fine_to_coarse.get(cls, "Unknown") for cls in classnames]
    coarse_classes = sorted(set(fine_to_coarse.values())) # Get unique coarse classes
    coarse_to_index = {cls: idx for idx, cls in enumerate(coarse_classes)}
    
    return train_dataset, val_dataset, test_dataset, classnames, coarse_labels, fine_to_coarse, coarse_to_index