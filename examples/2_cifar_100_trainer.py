import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import wandb
import os
import time
from argparse import Namespace

from gaf import train_GAF

# Ensure to set your WandB API key as an environment variable or directly in the code
# os.environ["WANDB_API_KEY"] = "your_wandb_api_key_here"

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Check for available device (GPU or CPU)
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    device_index = random.randint(0, num_gpus - 1)  # Pick a random device index
    device = torch.device(f"cuda:{device_index}")
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
  # Set up WandB project and run names
  model_name = 'ResNet18'
  dataset_name = 'CIFAR100'
  project_name = f"{model_name}_{dataset_name}_FLIPPED_LABELS_COSINE_SIM"
  run_name = f"example_run"
  wandb.init(project=project_name, name=run_name, config=args)


train_GAF(model,
          args,
          train_dataset,
          val_dataset,
          optimizer,
          criterion,
          wandb=args.wandb,
          verbose=args.verbose,
          cos_distance_thresh=args.cos_distance_thresh, 
          device=device)
