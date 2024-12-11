"""
Implementation of GAF in two methods.

Method 1: step_GAF()
Usage:
    To be used inside a train loop in lieu of optimizer.step()

Method 2: train_GAF()
Usage:
    To be used inside as the train loop in lieu of huggingface's trainer.train()

For more information, see documentation below.

Author:
    Francois Chaubard 

Date:
    2024-12-03
"""

import torch
from torch.utils.data import DataLoader, Subset
import random

def _filter_gradients_cosine_sim(G1, G2, cos_distance_thresh):
    """
    Filters gradients based on cosine similarity.

    Args:
        G1 (list[torch.Tensor]): List of parameter gradients from the first microbatch.
        G2 (list[torch.Tensor]): List of parameter gradients from the second microbatch.
        cos_distance_thresh (float): Threshold for cosine distance.

    Returns:
        tuple: (filtered_grad, cos_distance)
            filtered_grad (list[torch.Tensor] or None): The averaged gradients if cosine distance 
            is below threshold, otherwise None.
            cos_distance (float): The computed cosine distance between G1 and G2.
    """
    # Flatten G1 and G2 into vectors
    G1_flat = torch.cat([g1.view(-1) for g1 in G1])
    G2_flat = torch.cat([g2.view(-1) for g2 in G2])

    # Compute cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(G1_flat, G2_flat, dim=0)

    # Compute cosine distance
    cos_distance = 1 - cos_sim

    if cos_distance > cos_distance_thresh:
        filtered_grad = None
    else:
        filtered_grad = [(g1 + g2) / 2 for g1, g2 in zip(G1, G2)]

    return filtered_grad, cos_distance.item()

def _compute_gradients(b, optimizer, model, criterion, device):
    """
    Computes gradients for a given microbatch.

    Args:
        b (Subset): The microbatch dataset.
        optimizer (torch.optim.Optimizer): The optimizer.
        model (torch.nn.Module): The model.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to use.

    Returns:
        tuple: (G, loss, labels, outputs)
            G (list[torch.Tensor]): List of gradients for each model parameter.
            loss (torch.Tensor): Computed loss for the batch.
            labels (torch.Tensor): Labels for the batch.
            outputs (torch.Tensor): Model outputs for the batch.
    """
    loader = DataLoader(b, batch_size=len(b), shuffle=False)
    data = next(iter(loader))
    images, labels = data[0].to(device), data[1].to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    G = [p.grad.clone() for p in model.parameters()]
    optimizer.zero_grad()
    return G, loss, labels, outputs

def step_GAF(model, 
             optimizer, 
             criterion, 
             list_of_microbatches,
             wandb=True,
             verbose=True,
             cos_distance_thresh=1.0,
             device=torch.device('cpu')):
    """
    Performs one Gradient Agreement Filtering (GAF) step given a list of microbatches.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        criterion (torch.nn.Module): The loss function.
        list_of_microbatches (list[Subset]): A list of data subsets (microbatches) for GAF.
        wandb (bool): Whether to log training metrics to Weights & Biases.
        verbose (bool): Whether to print debug information.
        cos_distance_thresh (float): Cosine distance threshold for filtering. This is \tau in the paper. Must be between 0 and 2. We recommend 0.9 to 1 for an HPP sweep.
        device (torch.device): Device on which to perform computation. TODO: You may want to distribute this across GPUs which we may implement later to be in parellel.

    Returns:
        dict: A dictionary containing loss, cosine_distance, and agreed_count.
    """
    model.train()
    total_loss = 0
    total_correct = 0
    
    # Check if wandb is imported
    if wandb:
        import os
        import wandb

        # Check if the WANDB_API key is in the environment variables
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if not wandb_api_key:
            raise RuntimeError("No WandB API key provided. Please set the 'WANDB_API' environment variable.")

                 
    # Compute gradients on the first microbatch
    G_current, loss, labels, outputs = _compute_gradients(list_of_microbatches[0], optimizer, model, criterion, device)
    
    # update total_loss
    total_loss += loss * labels.size(0)

    # update @1 accuracy
    _, predicted = torch.max(outputs.data, 1)
    correct_top1 = (predicted == labels).sum().item()
    total_correct += correct_top1
    
    # Fuse gradients with subsequent microbatches
    agreed_count = 0
    
    for i, mb in enumerate(list_of_microbatches[1:]):
        # compute microgradients and filter them based on cosine distance:
        G, loss_i, labels_i, outputs_i = _compute_gradients(mb, optimizer, model, criterion, device)
        G_current_temp, cosine_distance = _filter_gradients_cosine_sim(G_current, G, cos_distance_thresh)
        
        # update total_loss
        total_loss += loss_i * labels_i.size(0)

        # update @1 accuracy
        _, predicted = torch.max(outputs_i.data, 1)
        correct_top1 = (predicted == labels_i).sum().item()
        total_correct += correct_top1
        
        if verbose:
            print(f"GAF fuse iter {i}, Cosine Distance: {cosine_distance:.4f}")

        if G_current_temp is not None:
            G_current = G_current_temp
            agreed_count += 1

    # If at least one agreed, update params
    if agreed_count > 0:
        with torch.no_grad():
            for param, grad in zip(model.parameters(), G_current):
                param.grad = grad
        optimizer.step()

    # Compute metrics 
    total = labels.size(0) * len(list_of_microbatches)
    result = {'train_loss': total_loss.item(), 'cosine_distance': cosine_distance, 'agreed_count':agreed_count, 'train_accuracy':total_correct/total }
                 
    if verbose:
        print(result)

    # Log to wandb
    if wandb:
        import wandb
        wandb.log(result)

    return result

def train_GAF(model,
              args,
              train_dataset,
              val_dataset,
              optimizer,
              criterion,
              wandb=True,
              verbose=True,
              cos_distance_thresh=1.0,
              device=torch.device('cpu')):
    """
    Trains the model using Gradient Agreement Filtering (GAF) across multiple epochs.
    This mimics the HuggingFace Trainer interface.

    Args:
        model (torch.nn.Module): The model to train.
        args (object): A simple namespace or dictionary with training arguments. 
                       Expected fields: 
                       - args.epochs (int)
                       - args.batch_size (int)
                       - args.num_batches_to_force_agreement (int)
        train_dataset (torch.utils.data.Dataset): Training dataset.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer to be used.
        criterion (torch.nn): Loss function to be used. 
        wandb (bool): Whether to log metrics to Weights & Biases.
        verbose (bool): Whether to print progress.
        cos_distance_thresh (float): Cosine distance threshold for GAF filtering. This is \tau in the paper. Must be between 0 and 2. We recommend 0.9 to 1 for an HPP sweep.
        device (torch.device): Device on which to train on. TODO: Make this an array in future iters size at most num_batches_to_force_agreement to run them in parallel.

    Returns:
        None
    """
    model.to(device)

    # Check if wandb is imported
    if wandb:
        import os
        import wandb

        # Check if the WANDB_API key is in the environment variables
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if not wandb_api_key:
            raise RuntimeError("No WandB API key provided. Please set the 'WANDB_API' environment variable.")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # A simple function to sample multiple microbatches for GAF
    def sample_iid_mbs_for_GAF(full_dataset, batch_size, n):
        # For simplicity: just take n random subsets of size batch_size
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        # If not enough data for n batches, just cycle
        if len(indices) < batch_size * n:
            multiples = (batch_size * n) // len(indices) + 1
            indices = (indices * multiples)[:batch_size * n]
        batches = []
        for i in range(n):
            subset_indices = indices[i*batch_size:(i+1)*batch_size]
            batches.append(Subset(full_dataset, subset_indices))
        return batches

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        count = 0

        steps_per_epoch = len(train_dataset) // (args.batch_size * args.num_batches_to_force_agreement)

        for step in range(steps_per_epoch):
            # Sample microbatches
            mbs = sample_iid_mbs_for_GAF(train_dataset, args.batch_size, args.num_batches_to_force_agreement)
            result = step_GAF(model,
                              optimizer,
                              criterion,
                              list_of_microbatches=mbs,
                              wandb=wandb,
                              verbose=verbose,
                              cos_distance_thresh=cos_distance_thresh,
                              device=device)
            running_loss += result['train_loss']
            count += 1

        # Validation step
        model.eval()
        val_loss = 0.0
        val_count = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch[0].to(device), batch[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_count += 1
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.cpu())

        val_loss /= max(val_count, 1)

        val_accuracy = sum([pred == label for pred, label in zip(all_preds, all_labels)]) / len(all_preds)
        message = {'epoch': epoch+1, 'train_loss': running_loss/max(count,1), 'val_loss': val_loss, 'train_accuracy': result['train_accuracy'], 'val_accuracy': val_accuracy.item()}
        
        if verbose:
            print(message)
        # log to wandb
        if wandb:
            try:
                import wandb
                wandb.log(message)
            except Exception as e:
                print(f"Failed to log to wandb: {e}")

    return None
