import argparse
import yaml
import torch
import wandb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trainers.cocoop_trainer import CoCoOpTrainer
from models.zeroshot import ZeroShotEvaluator
from utils.data_utils import build_dataset
from utils.dist_utils import init_distributed_mode
from settings import WANDB_API_KEY

def load_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def update_config_for_dataset(config, args):
    batch_size = args.batch_size if args.batch_size else 16
    shots = args.shots if args.shots else config['DATASET'].get('SHOTS', 0)
    model_type = args.model_type if args.model_type else config['MODEL']['TYPE']
    args_name = f"vit-32-{args.dataset}-{model_type}-{shots}-shots-bs{batch_size}"
    config["DATASET"]["BATCH_SIZE"] = batch_size
    config["DATASET"]["NAME"] = args.dataset
    config["DATASET"]["SHOTS"] = shots
    config["MODEL"]["TYPE"] = model_type
    config["ARTIFACT"]["NAME"] = args_name
    config["WANDB"]["RUN_NAME"] = args_name
    config["TRAINER"]["DISTRIBUTED"] = args.gpus > 1
    return config

def main():
    parser = argparse.ArgumentParser(description="Prompt Learning with CoCoOp")
    parser.add_argument("--dataset", type=str, default="CIFAR", help="Name of the dataset")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--shots", type=int, default=0, help="Number of shots per class (0 for full dataset)")
    parser.add_argument("--model_type", type=str, default="fine_only", 
                        choices=['plain_fine', 'plain_fine_coarse', 'fine_only', 'fine_coarse', 'fine_coarse_concat_cross', 'fine_coarse_separate', 'fine_coarse_concat', 'fine_context'],
                        help="Model type")
    parser.add_argument("--test_only", action="store_true", help="Only run testing")
    args = parser.parse_args()
    print("Arguments received:", args)

    cfg = load_config("configs/default.yaml")
    cfg = update_config_for_dataset(cfg, args)
    print("Configuration loaded and updated:", cfg)

    if cfg["TRAINER"]["DISTRIBUTED"]:
        init_distributed_mode(cfg)

    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(project=cfg["WANDB"]["PROJECT"], name=cfg["WANDB"]["RUN_NAME"], config=cfg)

    train_dataset, val_dataset, test_dataset, classnames, coarse_labels, fine_to_coarse, coarse_to_index = build_dataset(cfg)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg["DATASET"]["BATCH_SIZE"], 
        shuffle=False, 
        num_workers=cfg["DATASET"]["NUM_WORKERS"],
        pin_memory=cfg["DATASET"]["PIN_MEMORY"]
    )

    if cfg["MODEL"]["TYPE"] in ['plain_fine', 'plain_fine_coarse']:
        print("Running zero-shot evaluation...")
        evaluator = ZeroShotEvaluator(cfg, classnames, coarse_labels, fine_to_coarse, coarse_to_index)
        test_accuracy = evaluator.evaluate(test_loader)
        print(f"Zero-shot test accuracy: {test_accuracy:.2f}%")
    else:
        trainer = CoCoOpTrainer(cfg, classnames, coarse_labels, fine_to_coarse, coarse_to_index)
        
        if not args.test_only:
            print("Training model...")
            trainer.train(train_dataset, val_dataset)
        
        print("Evaluating model on test set...")
        test_accuracy = trainer.test(test_loader)
        print(f"Final test accuracy: {test_accuracy:.2f}%")

    wandb.finish()

if __name__ == "__main__":
    main()