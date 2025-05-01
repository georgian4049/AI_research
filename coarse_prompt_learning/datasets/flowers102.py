# from torchvision.datasets import Flowers102
# from torchvision import transforms


# # Manually defined class names
# FLOWERS102_CLASSNAMES = [
#     "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
#     "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon",
#     "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower",
#     "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
#     "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
#     "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist", "mexican aster",
#     "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
#     "lenten rose", "barberton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
#     "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy",
#     "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
#     "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush",
#     "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
#     "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower",
#     "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
#     "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm",
#     "ball moss", "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
#     "blanket flower", "trumpet creeper", "blackberry lily", "common tulip", "wild rose"
# ]

# # Fine-to-coarse label mapping
# FINE_TO_COARSE = {
#     "pink primrose": "Primroses",
#     "hard-leaved pocket orchid": "Orchids",
#     "canterbury bells": "Bellflowers",
#     "sweet pea": "Legumes",
#     "english marigold": "Marigolds",
#     "tiger lily": "Lilies",
#     "moon orchid": "Orchids",
#     "bird of paradise": "Tropical",
#     "monkshood": "Buttercups",
#     "globe thistle": "Thistles",
#     "snapdragon": "Figworts",
#     "colt's foot": "Daisies",
#     "king protea": "Proteas",
#     "spear thistle": "Thistles",
#     "yellow iris": "Irises",
#     "globe-flower": "Buttercups",
#     "purple coneflower": "Daisies",
#     "peruvian lily": "Lilies",
#     "balloon flower": "Bellflowers",
#     "giant white arum lily": "Lilies",
#     "fire lily": "Lilies",
#     "pincushion flower": "Teasel Family",
#     "fritillary": "Lilies",
#     "red ginger": "Tropical",
#     "grape hyacinth": "Hyacinths",
#     "corn poppy": "Poppies",
#     "prince of wales feathers": "Amaranths",
#     "stemless gentian": "Gentians",
#     "artichoke": "Herbs",
#     "sweet william": "Carnations",
#     "carnation": "Carnations",
#     "garden phlox": "Phloxes",
#     "love in the mist": "Buttercups",
#     "mexican aster": "Asters",
#     "alpine sea holly": "Apiaceae Family",
#     "ruby-lipped cattleya": "Orchids",
#     "cape flower": "Proteas",
#     "great masterwort": "Umbellifers",
#     "siam tulip": "Tropical",
#     "lenten rose": "Buttercups",
#     "barberton daisy": "Daisies",
#     "daffodil": "Amaryllis Family",
#     "sword lily": "Irises",
#     "poinsettia": "Spurges",
#     "bolero deep blue": "Bellflowers",
#     "wallflower": "Brassicaceae Family",
#     "marigold": "Marigolds",
#     "buttercup": "Buttercups",
#     "oxeye daisy": "Daisies",
#     "common dandelion": "Daisies",
#     "petunia": "Solanaceae Family",
#     "wild pansy": "Violas",
#     "primula": "Primroses",
#     "sunflower": "Asters",
#     "pelargonium": "Geraniums",
#     "bishop of llandaff": "Dahlias",
#     "gaura": "Onagraceae Family",
#     "geranium": "Geraniums",
#     "orange dahlia": "Dahlias",
#     "pink-yellow dahlia": "Dahlias",
#     "cautleya spicata": "Zingiberaceae Family",
#     "japanese anemone": "Buttercups",
#     "black-eyed susan": "Asters",
#     "silverbush": "Proteas",
#     "californian poppy": "Poppies",
#     "osteospermum": "Asters",
#     "spring crocus": "Iridaceae Family",
#     "bearded iris": "Irises",
#     "windflower": "Buttercups",
#     "tree poppy": "Poppies",
#     "gazania": "Asters",
#     "azalea": "Rhododendrons",
#     "water lily": "Water Plants",
#     "rose": "Roses",
#     "thorn apple": "Solanaceae Family",
#     "morning glory": "Convolvulaceae Family",
#     "passion flower": "Passifloraceae Family",
#     "lotus": "Water Plants",
#     "toad lily": "Lilies",
#     "anthurium": "Araceae Family",
#     "frangipani": "Apocynaceae Family",
#     "clematis": "Buttercups",
#     "hibiscus": "Malvaceae Family",
#     "columbine": "Buttercups",
#     "desert-rose": "Apocynaceae Family",
#     "tree mallow": "Malvaceae Family",
#     "magnolia": "Magnolias",
#     "cyclamen": "Primroses",
#     "watercress": "Brassicaceae Family",
#     "canna lily": "Cannaceae Family",
#     "hippeastrum": "Amaryllis Family",
#     "bee balm": "Lamiaceae Family",
#     "ball moss": "Bromeliaceae Family",
#     "foxglove": "Plantaginaceae Family",
#     "bougainvillea": "Nyctaginaceae Family",
#     "camellia": "Theaceae Family",
#     "mallow": "Malvaceae Family",
#     "mexican petunia": "Acanthaceae Family",
#     "bromelia": "Bromeliaceae Family",
#     "blanket flower": "Asters",
#     "trumpet creeper": "Bignoniaceae Family",
#     "blackberry lily": "Iridaceae Family",
#     "common tulip": "Lilies",
#     "wild rose": "Roses"
# }


import torch
from torchvision.datasets import CIFAR100, EuroSAT, Flowers102
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

# Fine-to-coarse label mapping

def build_flowers102_dataset(data_path, transform=transform, shots=0):
    # Load Flowers102 datasets
    train_dataset = Flowers102(root=data_path, split="train", download=True, transform=transform)
    val_dataset = Flowers102(root=data_path, split="val", download=True, transform=transform)
    test_dataset = Flowers102(root=data_path, split="test", download=True, transform=transform)
    
    # Fine class names
    classnames = [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold",
        "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon",
        "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower",
        "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower",
        "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
        "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist", "mexican aster",
        "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
        "lenten rose", "barberton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue",
        "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy",
        "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
        "pink-yellow dahlia", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush",
        "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy",
        "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower",
        "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose",
        "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm",
        "ball moss", "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
        "blanket flower", "trumpet creeper", "blackberry lily", "common tulip", "wild rose"
    ]
    
    # Fine-to-coarse mapping
    fine_to_coarse = {
        "pink primrose": "Primroses", "hard-leaved pocket orchid": "Orchids", "canterbury bells": "Bellflowers",
        "sweet pea": "Legumes", "english marigold": "Marigolds", "tiger lily": "Lilies", "moon orchid": "Orchids",
        "bird of paradise": "Tropical", "monkshood": "Buttercups", "globe thistle": "Thistles", "snapdragon": "Figworts",
        "colt's foot": "Daisies", "king protea": "Proteas", "spear thistle": "Thistles", "yellow iris": "Irises",
        "globe-flower": "Buttercups", "purple coneflower": "Daisies", "peruvian lily": "Lilies",
        "balloon flower": "Bellflowers", "giant white arum lily": "Lilies", "fire lily": "Lilies",
        "pincushion flower": "Teasel Family", "fritillary": "Lilies", "red ginger": "Tropical",
        "grape hyacinth": "Hyacinths", "corn poppy": "Poppies", "prince of wales feathers": "Amaranths",
        "stemless gentian": "Gentians", "artichoke": "Herbs", "sweet william": "Carnations", "carnation": "Carnations",
        "garden phlox": "Phloxes", "love in the mist": "Buttercups", "mexican aster": "Asters",
        "alpine sea holly": "Apiaceae Family", "ruby-lipped cattleya": "Orchids", "cape flower": "Proteas",
        "great masterwort": "Umbellifers", "siam tulip": "Tropical", "lenten rose": "Buttercups",
        "barberton daisy": "Daisies", "daffodil": "Amaryllis Family", "sword lily": "Irises", "poinsettia": "Spurges",
        "bolero deep blue": "Bellflowers", "wallflower": "Brassicaceae Family", "marigold": "Marigolds",
        "buttercup": "Buttercups", "oxeye daisy": "Daisies", "common dandelion": "Daisies", "petunia": "Solanaceae Family",
        "wild pansy": "Violas", "primula": "Primroses", "sunflower": "Asters", "pelargonium": "Geraniums",
        "bishop of llandaff": "Dahlias", "gaura": "Onagraceae Family", "geranium": "Geraniums", "orange dahlia": "Dahlias",
        "pink-yellow dahlia": "Dahlias", "cautleya spicata": "Zingiberaceae Family", "japanese anemone": "Buttercups",
        "black-eyed susan": "Asters", "silverbush": "Proteas", "californian poppy": "Poppies", "osteospermum": "Asters",
        "spring crocus": "Iridaceae Family", "bearded iris": "Irises", "windflower": "Buttercups", "tree poppy": "Poppies",
        "gazania": "Asters", "azalea": "Rhododendrons", "water lily": "Water Plants", "rose": "Roses",
        "thorn apple": "Solanaceae Family", "morning glory": "Convolvulaceae Family", "passion flower": "Passifloraceae Family",
        "lotus": "Water Plants", "toad lily": "Lilies", "anthurium": "Araceae Family", "frangipani": "Apocynaceae Family",
        "clematis": "Buttercups", "hibiscus": "Malvaceae Family", "columbine": "Buttercups", "desert-rose": "Apocynaceae Family",
        "tree mallow": "Malvaceae Family", "magnolia": "Magnolias", "cyclamen": "Primroses", "watercress": "Brassicaceae Family",
        "canna lily": "Cannaceae Family", "hippeastrum": "Amaryllis Family", "bee balm": "Lamiaceae Family",
        "ball moss": "Bromeliaceae Family", "foxglove": "Plantaginaceae Family", "bougainvillea": "Nyctaginaceae Family",
        "camellia": "Theaceae Family", "mallow": "Malvaceae Family", "mexican petunia": "Acanthaceae Family",
        "bromelia": "Bromeliaceae Family", "blanket flower": "Asters", "trumpet creeper": "Bignoniaceae Family",
        "blackberry lily": "Iridaceae Family", "common tulip": "Lilies", "wild rose": "Roses"
    }
    coarse_labels = [fine_to_coarse[cls] for cls in classnames]
    coarse_classes = sorted(set(coarse_labels))
    coarse_to_index = {cls: idx for idx, cls in enumerate(coarse_classes)}
    
    # Few-shot sampling on training set
    if shots > 0:
        labels = train_dataset._labels
        train_indices = []
        for cls in range(len(classnames)):
            cls_indices = np.where(np.array(labels) == cls)[0]
            selected_indices = np.random.choice(cls_indices, size=min(shots, len(cls_indices)), replace=False)
            train_indices.extend(selected_indices)
        train_indices = np.array(train_indices)
        np.random.shuffle(train_indices)
        train_dataset = Subset(train_dataset, train_indices)
    
    return train_dataset, val_dataset, test_dataset, classnames, coarse_labels, fine_to_coarse, coarse_to_index