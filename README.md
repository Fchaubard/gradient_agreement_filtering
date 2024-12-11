# Gradient Agreement Filtering (GAF)

This package implements the Gradient Agreement Filtering (GAF) optimization algorithm. 

GAF is a novel optimization algorithm that improves gradient-based optimization by filtering out gradients of data batches that do not agree with each other and nearly eliminates the need for a validation set without risk of overfitting (even with noisy labels). It bolts on top of existing optimization procedures such as SGD, SGD with Nesterov momentum, Adam, AdamW, RMSProp, etc and outperforms in all cases. Full paper here:
```
TODO: Insert arxiv paper link.
```

## Repo Features

- Supports multiple optimizers: SGD, SGD with Nesterov momentum, Adam, AdamW, RMSProp.
- Implements Gradient Agreement Filtering based on cosine distance. 
- Allows for label noise injection by flipping a percentage of labels.
- Customizable hyperparameters via command-line arguments.
- Logging and tracking with Weights & Biases (wandb).


## Requirements

- Python 3.6 or higher
- PyTorch 1.7 or higher
- torchvision 0.8 or higher
- numpy
- wandb

## Installation


   ```bash
   git clone https://github.com/<insert your username>/gradient_agreement_filtering.git
   cd gradient_agreement_filtering
   pip install .
   ```

## Usage

We provide two ways to easily incorporate GAF into your existing training. 
1. **step_GAF():**
   If you want to use GAF inside your existing train loop, you can just replace your typical:

   ```
   optimizer.zero_grad()
   outputs = model(batch)
   loss = criterion(outputs, labels)
   loss.backward()
   optimizer.step()
   ```
   
   with one call to step_GAF() as per below:
   
   ```
   from gaf import step_GAF
   ...
   results = step_GAF(model, 
             optimizer, 
             criterion, 
             list_of_microbatches,
             wandb=True,
             verbose=True,
             cos_distance_thresh=0.97,
             device=gpu_device)
   ...
   ```
   
2. **train_GAF():**

   If you want to use GAF as the train loop, you can just replace your typical hugging face / keras style interface:

   ```
   trainer.Train()
   ```
   
   with one call to train_GAF() as per below:
   
   ```
   from gaf import train_GAF
   ...
   train_GAF(model,
              args,
              train_dataset,
              val_dataset,
              optimizer,
              criterion,
              wandb=True,
              verbose=True,
              cos_distance_thresh=0.97,
              device=gpu_device)
   ...
   ```
   
## Examples

### NOTE: running with wandb
For all of the scripts below, if you want to run with wandb, you can either fill in the:
```
python examples/1_cifar_100_train_loop_exposed.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --learning_rate 0.01 --momentum 0.9 --nesterov True --wandb True --verbose True --num_samples_per_class_per_batch 1 --num_batches_to_force_agreement 2 --label_error_percentage 0.15 --cos_distance_thresh 0.97
```
Or you can prepend any of the calls below with:
```
os.environ["WANDB_API_KEY"] = "<your-wandb-api-key>"
```
Or you can login on the system first then run the .py via:
```
wandb login <your-wandb-api-key>
```
Or you can run without it. Choice is yours.

Now please review the examples below.

### 1_cifar_100_train_loop_exposed.py

This file uses **step_GAF()** to train a ResNet18 model on the CIFAR-100 dataset using PyTorch with the ability to add noise to the labels to observe how GAF performs under noisy conditions. The code supports various optimizers and configurations, allowing you to experiment with different settings to understand the impact of GAF on model training.

Example call:
```
python examples/1_cifar_100_train_loop_exposed.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --learning_rate 0.01 --momentum 0.9 --nesterov True --wandb True --verbose True --num_samples_per_class_per_batch 1 --num_batches_to_force_agreement 2 --label_error_percentage 0.15 --cos_distance_thresh 0.97
```

### 2_cifar_100_trainer.py
This file uses **train_GAF()** to train a ResNet18 model on the CIFAR-100 dataset using PyTorch just to show how it works. 

Example call:
```
python examples/2_cifar_100_trainer.py 
```

### 3_cifar_100N_train_loop_exposed.py

This file uses **step_GAF()** to train a ResNet34 model on the CIFAR-100N-Fine dataset using PyTorch to observe how GAF performs under typical labeling noise. The code supports various optimizers and configurations, allowing you to experiment with different settings to understand the impact of GAF on model training.

Example call:
```
python examples/3_cifar_100N_Fine_train_loop_exposed.py --GAF True --optimizer "SGD+Nesterov+val_plateau"  --cifarn True --learning_rate 0.01 --momentum 0.9 --nesterov True --wandb True --verbose True --num_samples_per_class_per_batch 2 --num_batches_to_force_agreement 2 --cos_distance_thresh 0.97
```


## Acknowledgement

To cite this work, please use the following BibTeX entry:

```
TODO
```

# Citing GAF
```
Insert bibtex
```

## License

This package is licensed under the MIT license. See [LICENSE](LICENSE) for details.

