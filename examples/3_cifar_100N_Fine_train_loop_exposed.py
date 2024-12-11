"""
Script to train ResNet34-PreAct (as per the standard benchmark) on CIFAR-100N-Fine with Gradient Agreement Filtering (GAF) and various optimizers.

This script allows for experimentation with different optimizers and GAF settings. It supports label noise injection and logs metrics using Weights & Biases (wandb).

We expose the train loop to allow maximum flexibility. 

Usage:
    python examples/3_cifar_100N_Fine_train_loop_exposed.py [OPTIONS]

Example:
    python examples/3_cifar_100N_Fine_train_loop_exposed.py --GAF True --optimizer "SGD+Nesterov+val_plateau"  --cifarn True --learning_rate 0.01 --momentum 0.9 --nesterov True --wandb True --verbose True --num_samples_per_class_per_batch 2 --num_batches_to_force_agreement 2 --cos_distance_thresh 0.97

Author:
    Francois Chaubard 

Date:
    2024-12-03
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import numpy as np
import random
import os
import argparse
import time

# Try to import wandb
try:
    import wandb
except ImportError:
    wandb = None

from gradient_agreement_filtering import step_GAF

####################################################################################################

# (OPTIONAL) WANDB SETUP

# Ensure to set your WandB API key as an environment variable or directly in the code
# os.environ["WANDB_API_KEY"] = ""

####################################################################################################

# DEFINITIONS

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


def sample_iid_mbs_for_GAF(full_dataset, class_indices, n, m):
    """
    Samples 'n' independent minibatches, each containing an equal number of samples from each class, m.

    Args:
        full_dataset (Dataset): The full training dataset.
        class_indices (dict): A mapping from class labels to data indices.
        n (int): The number of microbatches to sample.
        m (int): The number of images per class to sample per microbatch.

    Returns:
        list: A list of Subsets representing the minibatches.
    """
    # Initialize a list to hold indices for each batch
    batch_indices_list = [[] for _ in range(n)]
    for clazz in class_indices:
        num_samples_per_class = m  # Adjust if you want more samples per class per batch
        total_samples_needed = num_samples_per_class * n
        available_indices = class_indices[clazz]
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


'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def PreResNet18(num_classes):
    return ResNet(PreActBlock, [2,2,2,2],num_classes=num_classes)

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2],num_classes=num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3],num_classes=num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3],num_classes=num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3],num_classes=num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3],num_classes=num_classes)

####################################################################################################

# MAIN TRAINING LOOP

if __name__ == '__main__':

    # Define the list of available optimizer types
    optimizer_types = ["SGD", "SGD+Nesterov", "SGD+Nesterov+val_plateau", "Adam", "AdamW", "RMSProp"]

    parser = argparse.ArgumentParser(description='Train ResNet18 on CIFAR-100 with various optimizers and GAF.')

    # General training parameters
    parser.add_argument('--GAF', type=str2bool, default=True, help='Enable Gradient Agreement Filtering (True or False)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay factor')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_val_epochs', type=int, default=2, help='Number of epochs between validation checks')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=optimizer_types, help='Optimizer type to use')
    parser.add_argument('--num_batches_to_force_agreement', type=int, default=2, help='Number of batches to compute gradients for agreement (must be > 1)')
    parser.add_argument('--epochs', type=int, default=10000, help='Total number of training epochs')
    parser.add_argument('--num_samples_per_class_per_batch', type=int, default=1, help='Number of samples per class per batch when using GAF')
    parser.add_argument('--cos_distance_thresh', type=float, default=1, help='Threshold for cosine distance in gradient agreement filtering. Tau in the paper.')
    parser.add_argument('--dummy', type=bool, default=False, help='if we should use dummy data or not')
    parser.add_argument('--cifarn', type=bool, default=True, help='if we should use CIFARN labels or not')
    parser.add_argument('--cifarn_noisy_data_file_path', type=str, default="./examples/CIFAR-100_human.pt", help='the path to the noisy labeling file')

    # Optimizer-specific parameters
    parser.add_argument('--momentum', type=float, default=0.0, help='Momentum factor for SGD and RMSProp optimizers')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='Use Nesterov momentum (True or False)')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='Betas for Adam and AdamW optimizers')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value for optimizers')
    parser.add_argument('--alpha', type=float, default=0.99, help='Alpha value for RMSProp optimizer')
    parser.add_argument('--centered', type=str2bool, default=False, help='Centered RMSProp (True or False)')
    parser.add_argument('--scheduler_patience', type=int, default=100, help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--scheduler_factor', type=int, default=0.1, help='Discount factor for ReduceLROnPlateau scheduler')

    # logging
    parser.add_argument('--wandb', type=str2bool, default=False, help='Log to wandb (True or False)')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Print out logs (True or False)')

    # Parse arguments
    args = parser.parse_args()
    config = vars(args)

    # Set unused optimizer-specific configs to 'NA'
    optimizer = config['optimizer']
    all_params = ['momentum', 'nesterov', 'betas', 'eps', 'alpha', 'centered', 'scheduler_patience', 'scheduler_factor']

    # Define which parameters are used by each optimizer
    optimizer_params = {
        'SGD': ['momentum', 'nesterov'],
        'SGD+Nesterov': ['momentum', 'nesterov'],
        'SGD+Nesterov+val_plateau': ['momentum', 'nesterov', 'scheduler_patience', 'scheduler_factor'],
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
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if config['verbose']:
        print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    ####


    # Data transformations
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


    if config['dummy']==True:
        print('using dummy data')
        train_dataset = datasets.FakeData(
            size=50000,  # Match CIFAR-100 training set size
            image_size=(3, 32, 32),  # Match CIFAR-100 image size
            num_classes=100,
            transform=transform_train,  # Use the same training transforms
            random_offset=random.randint(0, 1000000)
        )
        # Create fake test data
        test_dataset = datasets.FakeData(
            size=10000,  # Match CIFAR-100 test set size
            image_size=(3, 32, 32),
            num_classes=100,
            transform=transform_test,  # Use the same test transforms
            random_offset=random.randint(0, 1000000)
        )
        
            
    elif config['cifarn']==True:
        print('using CIFAR100N data')
        train_dataset = datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transform_train
        )
    
        # Load the noisy labels
        dd = torch.load(config['cifarn_noisy_data_file_path']) # this should be part of the repo
        noisy_label = dd['noisy_label']
        # Replace the training labels with noisy labels
        train_dataset.targets = noisy_label
        # Load CIFAR-100 test dataset (clean labels)
        test_dataset = datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transform_test
        )

    else:
        print('using CIFAR100 data')
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)


    # Create a mapping from class labels to indices for sampling
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)

    # Initialize the model (ResNet34) and move it to the device
    model = ResNet34(num_classes=100)
    model = model.to(device)

    # Define the loss function (CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()

    # Initialize the optimizer based on the selected type and parameters
    if config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                            momentum=config['momentum'],
                            weight_decay=config['weight_decay'],
                            nesterov=config['nesterov'])
    elif config['optimizer'] == 'SGD+Nesterov':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                            momentum=config['momentum'],
                            weight_decay=config['weight_decay'],
                            nesterov=True)
    elif config['optimizer'] == 'SGD+Nesterov+val_plateau':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'],
                            momentum=config['momentum'],
                            weight_decay=config['weight_decay'],
                            nesterov=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['scheduler_patience'],  factor=config['scheduler_factor'])
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                            betas=tuple(config['betas']),
                            eps=config['eps'],
                            weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                betas=tuple(config['betas']),
                                eps=config['eps'],
                                weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=config['learning_rate'],
                                alpha=config['alpha'],
                                eps=config['eps'],
                                weight_decay=config['weight_decay'],
                                momentum=config['momentum'],
                                centered=config['centered'])
    else:
        raise ValueError(f"Unsupported optimizer type: {config['optimizer']}")

    # Test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    if config['wandb']:
        # Raise an error if wandb is not installed
        if wandb is None:
            raise ImportError("wandb is not installed. Please install it using 'pip install wandb'.")

        # Set up WandB project and run names
        model_name = 'ResNet34'
        dataset_name = 'CIFAR100N-Fine'
        project_name = f"{model_name}_{dataset_name}"
        
        # Initialize WandB
        wandb.init(project=project_name, config=config)
        config = wandb.config

    # Create checkpoints directory
    checkpoint_dir = './checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_timestamp = time.strftime("%Y%m%d_%H%M%S")

    iteration = 0
    try:
        # Training loop
        for epoch in range(config['epochs']):
            model.train()
            running_loss = 0.0
            correct_top1 = 0
            total = 0
            i = 0

            # Calculate total iterations per epoch
            if config['GAF']:
                iterations_per_epoch = len(train_dataset) // (len(class_indices) * config['num_samples_per_class_per_batch'])
            else:
                iterations_per_epoch = len(train_dataset) // config['batch_size']

            while i < iterations_per_epoch:
                i += 1
                if config['GAF']:
                    # Sample microbatches for GAF
                    mbs = sample_iid_mbs_for_GAF(train_dataset, class_indices, config['num_batches_to_force_agreement'], config['num_samples_per_class_per_batch'])
                    # Run GAF to update the model
                    result = step_GAF(model, optimizer, criterion, mbs, use_wandb=config['wandb'], verbose=config['verbose'], device=device)

                    # Update metrics
                    running_loss += result['train_loss'] / (len(class_indices) * config['num_batches_to_force_agreement'])
                    total += 1
                    correct_top1 += result['train_accuracy']

                    if config['verbose']:
                        print(f'Epoch: {epoch:7d}, Iteration: {iteration:7d}, Train Loss: {result["train_loss"]:.9f}, Train Acc: {result["train_accuracy"]:.4f}, Costine Distance: {result["cosine_distance"]:.4f}, Agreement Count: {result["agreed_count"]:d}')

                    # Log to wandb
                    if config['wandb']:
                        wandb.log(result)

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
                    loss.backward()
                    optimizer.step()

                    # Update metrics
                    running_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct_top1 += (predicted == labels).sum().item()

                    # print for baseline
                    message = {'train_loss': loss.item(),
                               'train_accuracy': (predicted == labels).sum().item() / labels.size(0),
                               'iteration': iteration}

                    # Log metrics to wandb
                    if config['wandb']:
                        try:
                            wandb.log(message)
                        except Exception as e:
                            print(f"Failed to log to wandb: {e}")
                    if config['verbose']:
                        # Print formatted metrics
                        print(f'Epoch: {epoch:7d}, Iteration: {iteration:7d}, Train Loss: {message["train_loss"]:.9f}, Train Acc: {message["train_accuracy"]:.4f}')

                iteration += 1

            # Perform validation every num_val_epochs iterations
            if epoch % config['num_val_epochs'] == 0 and total > 0:
                # Compute training metrics
                train_loss = running_loss / total
                train_accuracy = correct_top1 / total

                # Evaluate on the validation/test set
                val_loss, val_accuracy = evaluate(model, test_loader, device)
                message = {'train_loss': train_loss,
                           'train_accuracy': train_accuracy,
                           'val_loss': val_loss,
                           'val_accuracy': val_accuracy,
                           'epoch': epoch,
                           'iteration': iteration}
                # Log metrics to wandb
                if config['wandb']:
                    try:
                        wandb.log(message)
                    except Exception as e:
                        print(f"Failed to log to wandb: {e}")

                # Print formatted metrics
                print(f'Epoch: {epoch:7d}, Train Loss: {train_loss:.9f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.9f}, Val Acc: {val_accuracy:.4f}')

                # Reset running metrics
                running_loss = 0.0
                correct_top1 = 0
                total = 0
                # Save the latest checkpoint
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"cifar_100n_{start_timestamp}_checkpoint_{timestamp}.pt"

                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                try:
                    torch.save(model.state_dict(), checkpoint_path)
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")

                # Adjust learning rate if scheduler is used
                if config['optimizer'] == 'SGD+Nesterov+val_plateau':
                    scheduler.step(val_loss)

    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"cifar_100n_{start_timestamp}_interrupt_{timestamp}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        try:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
        exit(0)
