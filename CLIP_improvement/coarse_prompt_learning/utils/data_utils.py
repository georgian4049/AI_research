from datasets.cifar import build_cifar100_dataset
from datasets.oxford_pets import build_oxford_pets_dataset
from datasets.flowers102 import build_flowers102_dataset
from datasets.dtd import build_dtd_dataset
from datasets.eurosat import build_eurosat_dataset
from datasets.caltech101 import build_caltech101_dataset
from datasets.oxford_pets import build_oxford_pets_dataset
from datasets.cifar100_small import build_cifar100_small_dataset

from torchvision import transforms

def get_transform():
    """Shared transform for all datasets."""
    return transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])
    
    

def build_dataset(cfg):
    dataset_name = cfg["DATASET"]["NAME"]
    data_path = "../../../../../../work/ML/shekhar/"
    transform = get_transform()
    shots = cfg['DATASET'].get('SHOTS', 0)

    if dataset_name == "CIFAR":
        return build_cifar100_dataset(f"{data_path}/cifar", transform, shots)
    elif dataset_name == "CIFAR100_small":
        return build_cifar100_small_dataset(f"{data_path}/cifar_small", transform)
    elif dataset_name == "CALTECH101":
        return build_caltech101_dataset(f"{data_path}/caltech101", transform)
    elif dataset_name == "DTD":
        return build_dtd_dataset(f"{data_path}/dtd", transform)
    elif dataset_name == "EUROSAT":
        return build_eurosat_dataset(f"{data_path}/eurosat", transform)
    elif dataset_name == "FLOWERS102":
        return build_flowers102_dataset(f"{data_path}/flowers102", transform)
    elif dataset_name == "OXFORD_PETS":
        return build_oxford_pets_dataset(f"{data_path}/oxford_pets", transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")