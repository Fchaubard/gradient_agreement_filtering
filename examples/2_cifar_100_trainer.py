"""
Script to train ResNet18 on CIFAR-100 with Gradient Agreement Filtering (GAF) with test_GAF().

Usage:
    python examples/2_cifar_100_trainer.py 

Author:
    Francois Chaubard 

Date:
    2024-12-03
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import os
import time
from argparse import Namespace

# Try to import wandb
try:
    import wandb
except ImportError:
    wandb = None

from gradient_agreement_filtering import train_GAF


# Ensure to set your WandB API key as an environment variable or directly in the code
# os.environ["WANDB_API_KEY"] = "your_wandb_api_key_here"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check for available device (GPU or CPU)
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    device_index = random.randint(0, num_gpus - 1)  # Pick a random device index
    device = torch.device(f"cuda:{device_index}")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# Data setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Model
model = models.resnet18(num_classes=100)

# Define args
args = Namespace(
    epochs=100,
    batch_size=128,
    num_batches_to_force_agreement=2,
    learning_rate=0.01,
    momentum = 0.9,
    weight_decay = 1e-2,
    cos_distance_thresh = 0.98,
    wandb = True,
    verbose = True
)

optimizer = optim.SGD(model.parameters(), 
                      lr=args.learning_rate,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay,
                      nesterov=True)

criterion = nn.CrossEntropyLoss()

if wandb:
    # Raise an error if wandb is not installed
    if wandb is None:
        raise ImportError("wandb is not installed. Please install it using 'pip install wandb'.")

    # Set up WandB project and run names
    model_name = 'ResNet18'
    dataset_name = 'CIFAR100'
    project_name = f"{model_name}_{dataset_name}"
    run_name = f"example_run"
    wandb.init(project=project_name, name=run_name, config=args)


try:
    train_GAF(model,
            args,
            train_dataset,
            val_dataset,
            optimizer,
            criterion,
            use_wandb=args.wandb,
            verbose=args.verbose,
            cos_distance_thresh=args.cos_distance_thresh, 
            device=device)
except KeyboardInterrupt:
    # Save the model if the training is interrupted
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    checkpoint_dir = './checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_name = f"cifar_100_{timestamp}.pt"

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    try:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")
    if wandb:
        wandb.finish()
        print('WandB run finished')
    print('Training interrupted')
