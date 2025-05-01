import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from models.custom_clip import CustomCLIP
from models.clip_model import load_clip_to_cpu
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.nn import functional as F
from pathlib import Path
import os
import numpy as np

class CoCoOpTrainer:
    def __init__(self, cfg, classnames, coarse_labels, fine_to_coarse, coarse_to_index):
        self.cfg = cfg
        self.classnames = classnames
        self.coarse_labels = coarse_labels
        self.fine_to_coarse = fine_to_coarse
        self.coarse_to_index = coarse_to_index
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0
        self.base_dir = Path('~/../../work/ML/shekhar/').expanduser().resolve()
        self._setup_storage_paths()
        
        # Early stopping parameters
        self.early_stopping = cfg.get("TRAINER", {}).get("EARLY_STOPPING", True)
        self.patience = cfg.get("TRAINER", {}).get("PATIENCE", 10)
        self.min_delta = cfg.get("TRAINER", {}).get("MIN_DELTA", 0.001)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopped = False
        
        # Plateau detection parameters
        self.val_loss_history = []
        self.moving_avg_window = cfg.get("TRAINER", {}).get("MOVING_AVG_WINDOW", 5)
        
        self.build_model()

    def _setup_storage_paths(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        os.environ['WANDB_DIR'] = str(self.base_dir)
        os.environ['WANDB_CACHE_DIR'] = str(self.base_dir)
        os.environ['WANDB_CONFIG_DIR'] = str(self.base_dir)
        self.checkpoint_dir = self.base_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / 'checkpoint.pth'
        self.best_model_path = self.checkpoint_dir / 'best_model.pth'

    def build_model(self):
        clip_model = load_clip_to_cpu(self.cfg, self.device)
        if self.cfg['MODEL']['PRECISION'] in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(self.cfg, self.classnames, sorted(set(self.coarse_labels)) if self.coarse_labels else [], self.fine_to_coarse, clip_model).to(self.device)

        if self.cfg["TRAINER"]["DISTRIBUTED"]:
            self.model = DDP(self.model, device_ids=[dist.get_rank()])

        # Freeze all parameters except prompt learner
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.optim = optim.AdamW(self.model.prompt_learner.parameters(), lr=self.cfg["TRAINER"]["LR"], weight_decay=0.01)
        self.sched = CosineAnnealingWarmRestarts(self.optim, T_0=10, T_mult=2)
        self.scaler = GradScaler() if self.cfg['MODEL']['PRECISION'] == "amp" else None
        self.load_artifact_checkpoint()

    def load_artifact_checkpoint(self):
        try:
            artifact_name = f"{self.cfg['ARTIFACT']['NAME']}:latest"
            
            print(f"Loading artifact: {artifact_name}")
            artifact = wandb.use_artifact(artifact_name, type='model')
            artifact_dir = artifact.download()
            checkpoint_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
            if checkpoint_files:
                checkpoint_path = os.path.join(artifact_dir, checkpoint_files[0])
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch'] + 1
                if 'best_val_loss' in checkpoint:
                    self.best_val_loss = checkpoint['best_val_loss']
                print(f"Resumed from epoch: {checkpoint['epoch']}")
        except Exception as e:
            print(f"No existing artifact found or error: {e}")

    def train(self, train_dataset, val_dataset):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.cfg["DATASET"]["BATCH_SIZE"], shuffle=True, num_workers=self.cfg["DATASET"]["NUM_WORKERS"]
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.cfg["DATASET"]["BATCH_SIZE"], shuffle=False, num_workers=self.cfg["DATASET"]["NUM_WORKERS"]
        )

        for epoch in range(self.start_epoch, self.cfg["TRAINER"]["EPOCHS"]):
            self.model.train()
            running_loss_fine = 0.0
            running_loss_coarse = 0.0
            running_total_loss = 0.0

            for batch_idx, (images, fine_labels) in enumerate(train_loader):
                images, fine_labels = images.to(self.device), fine_labels.to(self.device)
                coarse_labels = self.get_coarse_labels(fine_labels) if self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] else None

                total_loss, loss_fine, loss_coarse = self.forward_backward(images, fine_labels, coarse_labels)
                running_total_loss += total_loss
                running_loss_fine += loss_fine
                running_loss_coarse += loss_coarse

                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.cfg['TRAINER']['EPOCHS']}], Batch [{batch_idx+1}/{len(train_loader)}], "
                          f"Total Loss: {total_loss:.4f}, Fine Loss: {loss_fine:.4f}, Coarse Loss: {loss_coarse:.4f}")

            avg_total_loss = running_total_loss / len(train_loader)
            avg_loss_fine = running_loss_fine / len(train_loader)
            avg_loss_coarse = running_loss_coarse / len(train_loader)
            print(f"Epoch {epoch+1} completed. Avg Total Loss: {avg_total_loss:.4f}, "
                  f"Avg Fine Loss: {avg_loss_fine:.4f}, Avg Coarse Loss: {avg_loss_coarse:.4f}")
            wandb.log({
                "train_total_loss": avg_total_loss,
                "train_loss_fine": avg_loss_fine,
                "train_loss_coarse": avg_loss_coarse,
                "epoch": epoch+1
            })

            # Evaluate model and check for early stopping
            val_metrics = self.evaluate(val_loader, epoch)
            val_loss = val_metrics["eval_loss_fine"]

            # Add current validation loss to history
            self.val_loss_history.append(val_loss)
            wandb.log({"val_loss_history": self.val_loss_history, "epoch": epoch+1})

            # Check for early stopping using plateau detection
            if self.early_stopping:
                if val_loss < self.best_val_loss:
                    improvement = self.best_val_loss - val_loss
                    self.best_val_loss = val_loss
                    self.save_model_to_wandb(epoch, is_best=True)
                    print(f"New best model saved! Validation loss improved by {improvement:.6f}")
                
                if len(self.val_loss_history) >= self.moving_avg_window:
                    recent_losses = self.val_loss_history[-self.moving_avg_window:]
                    avg_loss = sum(recent_losses) / len(recent_losses)
                    std_loss = np.std(recent_losses)
                    wandb.log({
                        "plateau_detection_avg_loss": avg_loss,
                        "plateau_detection_std_loss": std_loss,
                        "epoch": epoch+1
                    })
                    if std_loss < self.min_delta:
                        self.patience_counter += 1
                        print(f"Plateau detected (std_dev: {std_loss:.6f} < {self.min_delta}). Patience: {self.patience_counter}/{self.patience}")
                    else:
                        if val_loss < avg_loss:
                            self.patience_counter = 0
                            print(f"Still seeing significant changes in validation loss (std_dev: {std_loss:.6f}). Patience counter reset.")
                        else:
                            self.patience_counter += 1
                            print(f"Not improving compared to recent average. Patience: {self.patience_counter}/{self.patience}")
                else:
                    if val_loss >= self.best_val_loss:
                        self.patience_counter += 1
                        print(f"Validation didn't improve. Patience: {self.patience_counter}/{self.patience}")
                    else:
                        self.patience_counter = 0
                
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs!")
                    self.early_stopped = True
                    wandb.log({"early_stopped": True, "stopped_epoch": epoch+1})
                    break

            if epoch % 15 == 0:
                self.save_model_to_wandb(epoch)

            self.sched.step()

        if self.early_stopping and os.path.exists(self.best_model_path):
            best_checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            print(f"Training complete. Loaded best model from epoch {best_checkpoint['epoch']}")
            return best_checkpoint['epoch']
        
        return self.cfg["TRAINER"]["EPOCHS"]

    def test(self, test_loader):
        self.model.eval()
        correct_fine = 0
        correct_coarse = 0
        total = 0
        test_loss_fine = 0
        test_loss_coarse = 0

        with torch.no_grad():
            for images, fine_labels in test_loader:
                images, fine_labels = images.to(self.device), fine_labels.to(self.device)
                coarse_labels = self.get_coarse_labels(fine_labels) if self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] else None

                output = self.model(images)
                fine_logits = output[0] if isinstance(output, tuple) else output
                coarse_logits = output[1] if isinstance(output, tuple) and len(output) > 1 else None

                loss_fine = F.cross_entropy(fine_logits, fine_labels, reduction='mean')
                test_loss_fine += loss_fine.item()
                preds_fine = fine_logits.argmax(dim=1)
                correct_fine += (preds_fine == fine_labels).sum().item()

                if coarse_logits is not None and self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context']:
                    loss_coarse = F.cross_entropy(coarse_logits, coarse_labels, reduction='mean')
                    test_loss_coarse += loss_coarse.item()
                    preds_coarse = coarse_logits.argmax(dim=1)
                    correct_coarse += (preds_coarse == coarse_labels).sum().item()
                elif self.cfg['MODEL']['TYPE'] in ['fine_coarse_concat', 'fine_coarse_concat_cross'] and coarse_labels is not None:
                    coarse_logits = torch.zeros(fine_logits.size(0), len(self.model.coarse_classnames), device=fine_logits.device)
                    for fine_idx, coarse_cls in enumerate(self.model.fine_to_coarse.values()):
                        coarse_idx = self.model.coarse_classnames.index(coarse_cls)
                        coarse_logits[:, coarse_idx] += fine_logits[:, fine_idx]
                    loss_coarse = F.cross_entropy(coarse_logits, coarse_labels, reduction='mean')
                    test_loss_coarse += loss_coarse.item()
                    preds_coarse = coarse_logits.argmax(dim=1)
                    correct_coarse += (preds_coarse == coarse_labels).sum().item()

                total += fine_labels.size(0)

        avg_loss_fine = test_loss_fine / len(test_loader)
        avg_loss_coarse = test_loss_coarse / len(test_loader) if self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] else 0.0
        accuracy_fine = 100 * correct_fine / total
        accuracy_coarse = 100 * correct_coarse / total if self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] else 0.0

        print(f"Test Loss Fine: {avg_loss_fine:.4f}, Test Accuracy Fine: {accuracy_fine:.2f}%")
        if self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross']:
            print(f"Test Loss Coarse: {avg_loss_coarse:.4f}, Test Accuracy Coarse: {accuracy_coarse:.2f}%")

        wandb.log({
            "test_loss_fine": avg_loss_fine,
            "test_accuracy_fine": accuracy_fine,
            "test_loss_coarse": avg_loss_coarse,
            "test_accuracy_coarse": accuracy_coarse
        })

        return accuracy_fine

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        correct_fine = 0
        correct_coarse = 0
        total = 0
        val_loss_fine = 0
        val_loss_coarse = 0

        with torch.no_grad():
            for images, fine_labels in val_loader:
                images, fine_labels = images.to(self.device), fine_labels.to(self.device)
                coarse_labels = self.get_coarse_labels(fine_labels) if self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] else None

                output = self.model(images)
                fine_logits = output[0] if isinstance(output, tuple) else output
                coarse_logits = output[1] if isinstance(output, tuple) and len(output) > 1 else None

                loss_fine = F.cross_entropy(fine_logits, fine_labels, reduction='mean')
                val_loss_fine += loss_fine.item()
                preds_fine = fine_logits.argmax(dim=1)
                correct_fine += (preds_fine == fine_labels).sum().item()

                if coarse_logits is not None and self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context']:
                    loss_coarse = F.cross_entropy(coarse_logits, coarse_labels, reduction='mean')
                    val_loss_coarse += loss_coarse.item()
                    preds_coarse = coarse_logits.argmax(dim=1)
                    correct_coarse += (preds_coarse == coarse_labels).sum().item()
                elif self.cfg['MODEL']['TYPE'] in ['fine_coarse_concat', 'fine_coarse_concat_cross'] and coarse_labels is not None:
                    coarse_logits = torch.zeros(fine_logits.size(0), len(self.model.coarse_classnames), device=fine_logits.device)
                    for fine_idx, coarse_cls in enumerate(self.model.fine_to_coarse.values()):
                        coarse_idx = self.model.coarse_classnames.index(coarse_cls)
                        coarse_logits[:, coarse_idx] += fine_logits[:, fine_idx]
                    loss_coarse = F.cross_entropy(coarse_logits, coarse_labels, reduction='mean')
                    val_loss_coarse += loss_coarse.item()
                    preds_coarse = coarse_logits.argmax(dim=1)
                    correct_coarse += (preds_coarse == coarse_labels).sum().item()

                total += fine_labels.size(0)

        avg_loss_fine = val_loss_fine / len(val_loader)
        avg_loss_coarse = val_loss_coarse / len(val_loader) if self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] else 0.0
        accuracy_fine = 100 * correct_fine / total
        accuracy_coarse = 100 * correct_coarse / total if self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross'] else 0.0

        print(f"Eval Loss Fine: {avg_loss_fine:.4f}, Eval Accuracy Fine: {accuracy_fine:.2f}%")
        if self.cfg['MODEL']['TYPE'] in ['fine_coarse_separate', 'fine_context', 'fine_coarse_concat', 'fine_coarse_concat_cross']:
            print(f"Eval Loss Coarse: {avg_loss_coarse:.4f}, Eval Accuracy Coarse: {accuracy_coarse:.2f}%")

        metrics = {
            "eval_loss_fine": avg_loss_fine,
            "eval_accuracy_fine": accuracy_fine,
            "eval_loss_coarse": avg_loss_coarse,
            "eval_accuracy_coarse": accuracy_coarse,
            "epoch": epoch+1
        }
        
        wandb.log(metrics)
        return metrics

    def forward_backward(self, images, fine_labels, coarse_labels):
        self.optim.zero_grad()

        if self.cfg['MODEL']['PRECISION'] == "amp":
            with autocast(device_type="cuda"):
                total_loss, loss_fine, loss_coarse = self.model(images, fine_labels, coarse_labels)
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optim)
            torch.nn.utils.clip_grad_norm_(self.model.prompt_learner.parameters(), max_norm=1.0)
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            total_loss, loss_fine, loss_coarse = self.model(images, fine_labels, coarse_labels)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.prompt_learner.parameters(), max_norm=1.0)
            self.optim.step()

        return (
            total_loss.item(),
            loss_fine.item(),
            loss_coarse if isinstance(loss_coarse, float) else loss_coarse.item()
        )

    def save_model_to_wandb(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter
        }
        
        save_path = self.best_model_path if is_best else self.checkpoint_path
        torch.save(checkpoint, save_path)

        if not is_best:
            artifact = wandb.Artifact(
                name=self.cfg['ARTIFACT']['NAME'],
                type="model",
                metadata={
                    "dataset": self.cfg["DATASET"]["NAME"],
                    "batch_size": self.cfg["DATASET"]["BATCH_SIZE"],
                    "shots": self.cfg["DATASET"]["SHOTS"],
                    "model_type": self.cfg["MODEL"]["TYPE"],
                    "epoch": epoch,
                    "best_val_loss": float(self.best_val_loss)
                }
            )
            artifact.add_file(self.checkpoint_path)
            wandb.log_artifact(artifact)

    def get_coarse_labels(self, fine_labels):
        if not self.fine_to_coarse or not self.coarse_to_index:
            return None
        return torch.tensor([self.coarse_to_index[self.fine_to_coarse[self.classnames[label.item()]]] for label in fine_labels]).to(self.device)