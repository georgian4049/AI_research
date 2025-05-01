import torch
from torchvision.datasets import Caltech101
from torchvision.transforms import Compose, ToTensor, ConvertImageDtype, Resize, Normalize
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def stratified_split(dataset, test_size=0.2):
    labels = [dataset[i][1] for i in range(len(dataset))]  # Extract labels
    train_indices, val_indices = train_test_split(
        range(len(dataset)), test_size=test_size, stratify=labels, random_state=42
    )
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def build_caltech101_dataset(data_path, transform):
    # Ensure transform includes conversion to RGB and normalization
    transform = Compose([
        Resize((224, 224)),  # Resize images to 224x224
        ToTensor(),  # Convert PIL image to tensor
        ConvertImageDtype(torch.float32),  # Ensure dtype is float32
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load the dataset
    dataset = Caltech101(root=data_path, download=True, transform=transform)

    # Split the dataset into train and validation sets
    train_dataset, val_dataset = stratified_split(dataset)

    # Get class names
    classnames = dataset.categories  # Directly access class names

    # Define fine-to-coarse mapping with improved hierarchy
    fine_to_coarse = {
        # Mammals
        'beaver': 'Mammals', 'cougar_body': 'Mammals', 'cougar_face': 'Mammals',
        'dalmatian': 'Mammals', 'elephant': 'Mammals', 'gerenuk': 'Mammals',
        'hedgehog': 'Mammals', 'kangaroo': 'Mammals', 'leopard': 'Mammals',
        'llama': 'Mammals', 'okapi': 'Mammals', 'panda': 'Mammals',
        'platypus': 'Mammals', 'rhino': 'Mammals', 'wild_cat': 'Mammals',
        'American Bulldog': 'Mammals',

        # Birds
        'flamingo': 'Birds', 'flamingo_head': 'Birds', 'ibis': 'Birds',
        'pigeon': 'Birds', 'rooster': 'Birds',

        # Reptiles
        'crocodile': 'Reptiles', 'crocodile_head': 'Reptiles',

        # Insects
        'ant': 'Insects', 'butterfly': 'Insects', 'dragonfly': 'Insects',
        'mayfly': 'Insects', 'scorpion': 'Insects', 'tick': 'Insects',

        # Aquatic Animals
        'crab': 'Aquatic Animals', 'crayfish': 'Aquatic Animals', 'dolphin': 'Aquatic Animals',
        'lobster': 'Aquatic Animals', 'nautilus': 'Aquatic Animals', 'octopus': 'Aquatic Animals',
        'sea_horse': 'Aquatic Animals', 'trilobite': 'Aquatic Animals',

        # Vehicles
        'Motorbikes': 'Vehicles', 'airplanes': 'Vehicles', 'car_side': 'Vehicles',
        'ferry': 'Vehicles', 'helicopter': 'Vehicles', 'ketch': 'Vehicles',
        'schooner': 'Vehicles',

        # Household Items
        'anchor': 'Household Items', 'barrel': 'Household Items', 'binocular': 'Household Items',
        'camera': 'Household Items', 'cannon': 'Household Items', 'ceiling_fan': 'Household Items',
        'cellphone': 'Household Items', 'chair': 'Household Items', 'chandelier': 'Household Items',
        'cup': 'Household Items', 'dollar_bill': 'Household Items', 'ewer': 'Household Items',
        'garfield': 'Household Items', 'headphone': 'Household Items', 'inline_skate': 'Household Items',
        'lamp': 'Household Items', 'laptop': 'Household Items', 'menorah': 'Household Items',
        'metronome': 'Household Items', 'minaret': 'Household Items', 'pagoda': 'Household Items',
        'pyramid': 'Household Items', 'revolver': 'Household Items', 'scissors': 'Household Items',
        'snoopy': 'Household Items', 'soccer_ball': 'Household Items', 'stapler': 'Household Items',
        'stop_sign': 'Household Items', 'umbrella': 'Household Items', 'watch': 'Household Items',
        'wheelchair': 'Household Items', 'windsor_chair': 'Household Items', 'wrench': 'Household Items',
        'yin_yang': 'Household Items',

        # Musical Instruments
        'accordion': 'Musical Instruments', 'bass': 'Musical Instruments',
        'electric_guitar': 'Musical Instruments', 'euphonium': 'Musical Instruments',
        'grand_piano': 'Musical Instruments', 'mandolin': 'Musical Instruments',
        'saxophone': 'Musical Instruments',

        # Food
        'pizza': 'Food', 'strawberry': 'Food',

        # Plants
        'bonsai': 'Plants', 'joshua_tree': 'Plants', 'lotus': 'Plants',
        'sunflower': 'Plants', 'water_lilly': 'Plants',

        # Faces
        'Faces': 'Faces', 'Faces_easy': 'Faces',

        # Miscellaneous
        'background': 'Miscellaneous'
    }

    # Map fine classes to coarse classes
    coarse_labels = [fine_to_coarse.get(cls, "Unknown") for cls in classnames]

    coarse_classes = sorted(set(fine_to_coarse.values()))  # Get unique coarse classes
    coarse_to_index = {cls: idx for idx, cls in enumerate(coarse_classes)}

    return train_dataset, val_dataset, classnames, coarse_classes, fine_to_coarse, coarse_to_index