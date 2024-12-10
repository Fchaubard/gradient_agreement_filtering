"""
Script to train ResNet18 on CIFAR-100 with Gradient Agreement Filtering (GAF) and various optimizers.

This script allows for experimentation with different optimizers and GAF settings. It supports label noise injection and logs metrics using Weights & Biases (wandb).

We expose the train loop to allow maximum flexibility. 

Usage:
    python 1_cifar_100_train_loop_exposed.py [OPTIONS]

Example:
    python 1_cifar_100_train_loop_exposed.py --GAF True --optimizer "SGD+Nesterov" --learning_rate 0.01 --momentum 0.9 --nesterov True

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
from collections import defaultdict
import numpy as np
import random
import wandb
import os
import argparse
from gaf import step_GAF
####################################################################################################





def str2bool(v):
    """Parse boolean values from the command line."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Ensure to set your WandB API key as an environment variable or directly in the code
# os.environ["WANDB_API_KEY"] = "your_wandb_api_key_here"

# Define the list of available optimizer types
optimizer_types = ["SGD", "SGD+Nesterov", "SGD+Nesterov+val_plateau", "Adam", "AdamW", "RMSProp"]

parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-100 with various optimizers and GAF.')

# General training parameters
parser.add_argument('--GAF', type=str2bool, default=True, help='Enable Gradient Agreement Filtering (True or False)')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay factor')
parser.add_argument('--weight_decay_type', type=str, default='l2', choices=['l1', 'l2'], help='Type of weight decay to apply (l1 or l2)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--num_val_epochs', type=int, default=2, help='Number of epochs between validation checks')
parser.add_argument('--min_grad', type=float, default=0.0, help='Minimum gradient value for filtering in GAF')
parser.add_argument('--epsilon', type=float, default=1e-1, help='Epsilon value for gradient agreement filtering')
parser.add_argument('--optimizer', type=str, default='SGD', choices=optimizer_types, help='Optimizer type to use')
parser.add_argument('--num_batches_to_force_agreement', type=int, default=10, help='Number of batches to compute gradients for agreement (must be > 1)')
parser.add_argument('--epochs', type=int, default=10000, help='Total number of training epochs')
parser.add_argument('--num_samples_per_class_per_batch', type=int, default=1, help='Number of samples per class per batch when using GAF')
parser.add_argument('--label_error_percentage', type=float, default=0, help='Percentage of labels to flip in the training dataset to simulate label noise (between 0 and 1)')
parser.add_argument('--cos_distance_thresh', type=float, default=1, help='Threshold for cosine distance in gradient agreement filtering. Tau in the paper.')

# Optimizer-specific parameters
parser.add_argument('--momentum', type=float, default=0.0, help='Momentum factor for SGD and RMSProp optimizers')
parser.add_argument('--nesterov', type=str2bool, default=False, help='Use Nesterov momentum (True or False)')
parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='Betas for Adam and AdamW optimizers')
parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value for optimizers')
parser.add_argument('--alpha', type=float, default=0.99, help='Alpha value for RMSProp optimizer')
parser.add_argument('--centered', type=str2bool, default=False, help='Centered RMSProp (True or False)')
parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for ReduceLROnPlateau scheduler')

# logging
parser.add_argument('--wandb', type=str2bool, default=False, help='Log to wandb (True or False)')
parser.add_argument('--verbose', type=str2bool, default=False, help='Print out logs (True or False)')

# Parse arguments
args = parser.parse_args()
config = vars(args)

# Set unused optimizer-specific configs to 'NA'
optimizer = config['optimizer']
all_params = ['momentum', 'nesterov', 'betas', 'eps', 'alpha', 'centered', 'scheduler_patience']

# Define which parameters are used by each optimizer
optimizer_params = {
    'SGD': ['momentum', 'nesterov'],
    'SGD+Nesterov': ['momentum', 'nesterov'],
    'SGD+Nesterov+val_plateau': ['momentum', 'nesterov', 'scheduler_patience'],
    'Adam': ['betas', 'eps'],
    'AdamW': ['betas', 'eps'],
    'RMSProp': ['momentum', 'eps', 'alpha', 'centered'],
}

# Get the list of parameters used by the selected optimizer
used_params = optimizer_params.get(optimizer, [])

# Set unused parameters to 'NA'
for param in all_params:
    if param not in used_params:
        config[param] = 'NA'

# Check for available device (GPU or CPU)
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    device_index = random.randint(0, num_gpus - 1)  # Pick a random device index
    device = torch.device(f"cuda:{device_index}")
else:
    device = torch.device("cpu")

if config['verbose']:
  print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Data transformations for training and testing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408),
                         (0.2675, 0.2565, 0.2761)),
])

# Load CIFAR-100 dataset
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

# Create a mapping from class labels to indices for sampling
class_indices = defaultdict(list)
for idx, (_, label) in enumerate(train_dataset):
    class_indices[label].append(idx)

def sample_iid_mbs(full_dataset, class_indices, batch_size):
    """
    Samples an IID minibatch for standard training.

    Args:
        full_dataset (Dataset): The full training dataset.
        class_indices (dict): A mapping from class labels to data indices.
        batch_size (int): The size of the batch to sample.

    Returns:
        Subset: A subset of the dataset representing the minibatch.
    """
    num_classes = len(class_indices)
    samples_per_class = batch_size // num_classes
    batch_indices = []
    for cls in class_indices:
        indices = random.sample(class_indices[cls], samples_per_class)
        batch_indices.extend(indices)
    # If batch_size is not divisible by num_classes, fill the rest randomly
    remaining = batch_size - len(batch_indices)
    if remaining > 0:
        all_indices = [idx for idx in range(len(full_dataset))]
        batch_indices.extend(random.sample(all_indices, remaining))
    # Create a Subset
    batch = Subset(full_dataset, batch_indices)
    return batch

def flip_labels(train_dataset, label_error_percentage=0.1, num_classes=100):
    """
    Flips a percentage of labels in the training dataset to simulate label noise.

    Args:
        train_dataset (Dataset): The training dataset.
        label_error_percentage (float): The percentage of labels to flip (between 0 and 1).
        num_classes (int): The total number of classes.

    Returns:
        Dataset: The training dataset with labels flipped.
    """
    num_samples = len(train_dataset.targets)
    num_to_flip = int(label_error_percentage * num_samples)
    all_indices = list(range(num_samples))
    flip_indices = random.sample(all_indices, num_to_flip)

    for idx in flip_indices:
        original_label = train_dataset.targets[idx]
        # Exclude the original label to ensure the label is actually changed
        wrong_labels = list(range(num_classes))
        wrong_labels.remove(original_label)
        new_label = random.choice(wrong_labels)
        train_dataset.targets[idx] = new_label

    return train_dataset

def sample_iid_mbs_for_GAF(full_dataset, class_indices, n):
    """
    Samples 'n' independent minibatches, each containing an equal number of samples from each class.

    Args:
        full_dataset (Dataset): The full training dataset.
        class_indices (dict): A mapping from class labels to data indices.
        n (int): The number of minibatches to sample.

    Returns:
        list: A list of Subsets representing the minibatches.
    """
    # Initialize a list to hold indices for each batch
    batch_indices_list = [[] for _ in range(n)]
    for cls in class_indices:
        num_samples_per_class = 1  # Adjust if you want more samples per class per batch
        total_samples_needed = num_samples_per_class * n
        available_indices = class_indices[cls]
        # Ensure there are enough indices
        if len(available_indices) < total_samples_needed:
            multiples = (total_samples_needed // len(available_indices)) + 1
            extended_indices = (available_indices * multiples)[:total_samples_needed]
        else:
            extended_indices = random.sample(available_indices, total_samples_needed)
        for i in range(n):
            start_idx = i * num_samples_per_class
            end_idx = start_idx + num_samples_per_class
            batch_indices_list[i].extend(extended_indices[start_idx:end_idx])
    # Create Subsets for each batch
    batches = [Subset(full_dataset, batch_indices) for batch_indices in batch_indices_list]
    return batches

# Initialize the model (ResNet18) and move it to the device
model = models.resnet18(num_classes=100)
model = model.to(device)

# Define the loss function (CrossEntropyLoss)
criterion = nn.CrossEntropyLoss()

# Handle weight decay and L1 regularization
if config['weight_decay_type'] == 'l1':
    weight_decay = 0.0
    l1_lambda = config['weight_decay']
elif config['weight_decay_type'] == 'l2':
    weight_decay = config['weight_decay']
    l1_lambda = 0.0
else:
    raise ValueError("weight_decay_type must be 'l1' or 'l2'")

# Initialize the optimizer based on the selected type and parameters
if config['optimizer'] == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'],
                          weight_decay=weight_decay,
                          nesterov=config['nesterov'])
elif config['optimizer'] == 'SGD+Nesterov':
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'],
                          weight_decay=weight_decay,
                          nesterov=True)
elif config['optimizer'] == 'SGD+Nesterov+val_plateau':
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                          momentum=config['momentum'],
                          weight_decay=weight_decay,
                          nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['scheduler_patience'])
elif config['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                           betas=tuple(config['betas']),
                           eps=config['eps'],
                           weight_decay=weight_decay)
elif config['optimizer'] == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            betas=tuple(config['betas']),
                            eps=config['eps'],
                            weight_decay=weight_decay)
elif config['optimizer'] == 'RMSProp':
    optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'],
                              alpha=config['alpha'],
                              eps=config['eps'],
                              weight_decay=weight_decay,
                              momentum=config['momentum'],
                              centered=config['centered'])
else:
    raise ValueError(f"Unsupported optimizer type: {config['optimizer']}")

def evaluate(model, dataloader, device):
    """
    Evaluates the model on the validation or test dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The DataLoader for the dataset.
        device (torch.device): The device to use.

    Returns:
        tuple: Average loss and top-1 accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy_top1 = correct_top1 / total
    return avg_loss, accuracy_top1

# Test DataLoader
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

if config['wandb']:
  # Set up WandB project and run names
  model_name = 'ResNet18'
  dataset_name = 'CIFAR100'
  project_name = f"{model_name}_{dataset_name}_FLIPPED_LABELS_COSINE_SIM"
  name_prefix = 'GAF' if config['GAF'] else 'NO_GAF'
  run_name = f"{name_prefix}_opt_{config['optimizer']}_lr_{config['learning_rate']}_bs_{config['batch_size']}"
  
  # Initialize WandB
  wandb.init(project=project_name, name=run_name, config=config)
  config = wandb.config

# Create checkpoints directory
checkpoint_dir = './checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)

# If label error percentage is specified, flip labels to introduce label noise
if config['label_error_percentage'] and config['label_error_percentage'] > 0:
    if 0 < config['label_error_percentage'] < 1:
        train_dataset = flip_labels(train_dataset, label_error_percentage=config['label_error_percentage'], num_classes=len(class_indices))
    else:
        raise ValueError(f"label_error_percentage needs to be between 0 and 1. Given label_error_percentage={config['label_error_percentage']}")

iteration = 0
# Training loop
for epoch in range(config['epochs']):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    total = 0
    
    # Calculate total iterations per epoch
    if config['GAF']:
        iterations_per_epoch = len(train_dataset) // (len(class_indices) * config['num_samples_per_class_per_batch'])
    else:
        iterations_per_epoch = len(train_dataset) // config['batch_size']

    while iteration < iterations_per_epoch:
        if config['GAF']:
            # Sample microbatches for GAF
            mbs = sample_iid_mbs_for_GAF(train_dataset, class_indices, config['num_batches_to_force_agreement'])
            # Run GAF to update the model
            result = step_GAF(model, optimizer, criterion, mbs, compute_metrics=compute_metrics, wandb=config['wandb'], verbose=config['verbose'], device=device)
            
      
            if config['verbose']:
               print(result)

            # # Update metrics
            running_loss += result['loss'].item()
            total +=  len(class_indices)*config['num_batches_to_force_agreement']
            correct_top1 += result['correct_top1'] 

        else:
            # Sample a minibatch for standard training
            batch = sample_iid_mbs(train_dataset, class_indices, config['batch_size'])
            loader = DataLoader(batch, batch_size=len(batch), shuffle=False)
            data = next(iter(loader))
            images, labels = data[0].to(device), data[1].to(device)
            # Forward and backward passes
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            # L1 regularization if specified
            if config['weight_decay_type'] == 'l1':
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()
            # Update metrics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()

        iteration += 1

    # Perform validation every num_val_epochs iterations
    if epoch % config['num_val_epochs'] == 0 and total > 0:
        # Compute training metrics
        train_loss = running_loss / total
        train_accuracy = correct_top1 / total
      
        # Evaluate on the validation/test set
        val_loss, val_accuracy = evaluate(model, test_loader, device)
        
        # Log metrics to wandb
        if config['wandb']:
          try:
              wandb.log({
                  'train_loss': train_loss,
                  'train_accuracy': train_accuracy,
                  'val_loss': val_loss,
                  'val_accuracy': val_accuracy,
                  'epoch': epoch,
                  'iteration': iteration,
              })
          except Exception as e:
              print(f"Failed to log to wandb: {e}")
        if config['verbose']:
          print(f"Epoch [{epoch+1}/{config['epochs']}], Iteration [{iteration}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy*100:.2f}%")
        # Reset running metrics
        running_loss = 0.0
        correct_top1 = 0
        total = 0
        # Save the latest checkpoint
        checkpoint_name = f"{run_name}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        try:
            torch.save(model.state_dict(), checkpoint_path)
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        # Adjust learning rate if scheduler is used
        if config['optimizer'] == 'SGD+Nesterov+val_plateau':
            scheduler.step(val_loss)
