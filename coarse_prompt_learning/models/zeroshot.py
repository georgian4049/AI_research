import clip
import torch
from torchvision import transforms
from tqdm import tqdm
import wandb
from torch.nn import functional as F

class ZeroShotEvaluator:
    def __init__(self, cfg, classnames, coarse_labels, fine_to_coarse, coarse_to_index):
        self.cfg = cfg
        self.classnames = classnames
        self.coarse_labels = coarse_labels
        self.fine_to_coarse = fine_to_coarse
        self.coarse_to_index = coarse_to_index
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = cfg['MODEL']['TYPE']
        self.dataset = cfg['DATASET']['NAME'].lower()
        
        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Initialize WandB logging
        wandb.config.update({
            "model": "ViT-B/32",
            "prompt_type": "hierarchical" if self.model_type == "plain_fine_coarse" else "simple",
            "dataset": self.dataset
        })

    def generate_prompts(self):
        # Generate prompts based on model type
        if self.model_type == 'plain_fine_coarse':
            # Check if coarse labels and mapping are available
            if not self.coarse_labels or not self.fine_to_coarse:
                raise ValueError(f"Coarse labels or fine-to-coarse mapping not provided for {self.dataset} with plain_fine_coarse")
            prompts = [
                f"A photo of a {self.classnames[i]} of type {self.coarse_labels[i]}."
                for i in range(len(self.classnames))
            ]
        else:  # plain_fine
            prompts = [f"A photo of a {self.classnames[i]}." for i in range(len(self.classnames))]
        
        # Tokenize prompts
        text_inputs = clip.tokenize(prompts).to(self.device)
        return text_inputs, prompts

    def evaluate(self, test_loader):
        self.model.eval()
        text_inputs, prompts = self.generate_prompts()
        
        correct_fine = 0
        correct_coarse = 0
        total = 0
        coarse_classes = sorted(set(self.coarse_labels)) if self.coarse_labels else []
        
        with torch.no_grad():
            for images, fine_labels in tqdm(test_loader, desc=f"Zero-shot evaluation on {self.dataset}"):
                images, fine_labels = images.to(self.device), fine_labels.to(self.device)
                
                # Compute coarse labels if available
                coarse_labels = None
                if self.coarse_labels and self.fine_to_coarse:
                    coarse_labels = torch.tensor([
                        self.coarse_to_index[self.fine_to_coarse[self.classnames[label.item()]]]
                        for label in fine_labels
                    ]).to(self.device)

                # Encode images
                image_features = self.model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Encode text
                text_features = self.model.encode_text(text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Compute fine-grained logits
                fine_logits = (image_features @ text_features.T) * self.model.logit_scale.exp()
                preds_fine = fine_logits.argmax(dim=1)
                correct_fine += (preds_fine == fine_labels).sum().item()
                
                # Compute coarse-grained accuracy if coarse labels are available
                if coarse_labels is not None:
                    preds_coarse = torch.tensor([
                        self.coarse_to_index[self.fine_to_coarse[self.classnames[pred.item()]]]
                        for pred in preds_fine
                    ]).to(self.device)
                    correct_coarse += (preds_coarse == coarse_labels).sum().item()
                
                total += fine_labels.size(0)
        
        accuracy_fine = 100 * correct_fine / total
        accuracy_coarse = 100 * correct_coarse / total if coarse_labels is not None else 0.0
        
        print(f"Zero-shot Fine Accuracy on {self.dataset.upper()}: {accuracy_fine:.2f}%")
        if coarse_labels is not None:
            print(f"Zero-shot Coarse Accuracy on {self.dataset.upper()}: {accuracy_coarse:.2f}%")
        
        # Log to WandB
        wandb.log({
            "test_accuracy_fine": accuracy_fine,
            "test_accuracy_coarse": accuracy_coarse if coarse_labels is not None else None,
            "num_samples": total
        })
        
        return accuracy_fine