from torchvision.datasets import OxfordIIITPet
from torchvision import transforms


def build_oxford_pets_dataset(data_path, transform):
    _transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 for better cropping
        transforms.RandomCrop((224, 224)),  # Crop randomly for better generalization
        transforms.RandomHorizontalFlip(p=0.5),  # Helps with left/right variance
        transforms.RandomRotation(degrees=20),  # Small rotations help with pose variations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Simulates lighting changes
        transforms.GaussianBlur(kernel_size=3),  # Helps with texture-based biases
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization (ImageNet)
    ])
    train_dataset = OxfordIIITPet(root=data_path, split='trainval', download=True, transform=_transform)
    val_dataset = OxfordIIITPet(root=data_path, split='test', download=True, transform=_transform)

    classnames = train_dataset.classes
    # Define fine-to-coarse mapping manually
    # fine_to_coarse = {
    # 'Abyssinian': 'Cats',
    # 'Bengal': 'Cats',
    # 'Birman': 'Cats',
    # 'Bombay': 'Cats',
    # 'British Shorthair': 'Cats',
    # 'Egyptian Mau': 'Cats',
    # 'Maine Coon': 'Cats',
    # 'Persian': 'Cats',
    # 'Ragdoll': 'Cats',
    # 'Russian Blue': 'Cats',
    # 'Siamese': 'Cats',
    # 'Sphynx': 'Cats',
    # 'American Bulldog': 'Dogs',
    # 'American Pit Bull Terrier': 'Dogs',
    # 'Basset Hound': 'Dogs',
    # 'Beagle': 'Dogs',
    # 'Boxer': 'Dogs',
    # 'Chihuahua': 'Dogs',
    # 'English Cocker Spaniel': 'Dogs',
    # 'English Setter': 'Dogs',
    # 'German Shorthaired': 'Dogs',
    # 'Great Pyrenees': 'Dogs',
    # 'Havanese': 'Dogs',
    # 'Japanese Chin': 'Dogs',
    # 'Keeshond': 'Dogs',
    # 'Leonberger': 'Dogs',
    # 'Miniature Pinscher': 'Dogs',
    # 'Newfoundland': 'Dogs',
    # 'Pomeranian': 'Dogs',
    # 'Pug': 'Dogs',
    # 'Saint Bernard': 'Dogs',
    # 'Samoyed': 'Dogs',
    # 'Scottish Terrier': 'Dogs',
    # 'Shiba Inu': 'Dogs',
    # 'Staffordshire Bull Terrier': 'Dogs',
    # 'Wheaten Terrier': 'Dogs',
    # 'Yorkshire Terrier': 'Dogs'
    # }
    
    fine_to_coarse = {
        # Short-haired Cats
        'Abyssinian': 'Short-haired Cats',
        'Bengal': 'Short-haired Cats',
        'Bombay': 'Short-haired Cats',
        'Egyptian Mau': 'Short-haired Cats',
        'Russian Blue': 'Short-haired Cats',
        'Siamese': 'Short-haired Cats',
        'Sphynx': 'Short-haired Cats',
        
        # Long-haired Cats
        'Birman': 'Long-haired Cats',
        'British Shorthair': 'Long-haired Cats',
        'Maine Coon': 'Long-haired Cats',
        'Persian': 'Long-haired Cats',
        'Ragdoll': 'Long-haired Cats',

        # Toy Breeds
        'Chihuahua': 'Toy Breeds',
        'Havanese': 'Toy Breeds',
        'Japanese Chin': 'Toy Breeds',
        'Pomeranian': 'Toy Breeds',
        'Pug': 'Toy Breeds',
        'Yorkshire Terrier': 'Toy Breeds',
        
        # Hounds
        'Basset Hound': 'Hounds',
        'Beagle': 'Hounds',

        # Working Dogs
        'Boxer': 'Working Dogs',
        'Great Pyrenees': 'Working Dogs',
        'Newfoundland': 'Working Dogs',
        'Saint Bernard': 'Working Dogs',
        'Samoyed': 'Working Dogs',

        # Terrier Breeds
        'Scottish Terrier': 'Terrier Breeds',
        'Staffordshire Bull Terrier': 'Terrier Breeds',
        'Wheaten Terrier': 'Terrier Breeds',

        # Sporting Dogs
        'English Cocker Spaniel': 'Sporting Dogs',
        'English Setter': 'Sporting Dogs',
        'German Shorthaired': 'Sporting Dogs',
        'Leonberger': 'Sporting Dogs',

        # Non-Sporting Dogs
        'American Bulldog': 'Non-Sporting Dogs',
        'American Pit Bull Terrier': 'Non-Sporting Dogs',
        'Keeshond': 'Non-Sporting Dogs',
        'Miniature Pinscher': 'Non-Sporting Dogs',

        # Herding Dogs
        'Shiba Inu': 'Herding Dogs'
    }

    # coarse_labels = [fine_to_coarse[cls] for cls in classnames]
    coarse_labels = [fine_to_coarse.get(cls, "Unknown") for cls in classnames]
    
    coarse_classes = sorted(set(fine_to_coarse.values()))  # Get unique coarse classes
    print("oxford pets coarse classes: ", coarse_classes)
    coarse_to_index = {cls: idx for idx, cls in enumerate(coarse_classes)}

    return train_dataset, val_dataset, classnames, coarse_labels, fine_to_coarse, coarse_to_index