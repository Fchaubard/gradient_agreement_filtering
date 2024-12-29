# Gradient Agreement Filtering (GAF)

This package implements the Gradient Agreement Filtering (GAF) optimization algorithm. 

GAF is a novel optimization algorithm that improves gradient-based optimization by filtering out gradients of data batches that do not agree with each other and nearly eliminates the need for a validation set without risk of overfitting (even with noisy labels). It bolts on top of existing optimization procedures such as SGD, SGD with Nesterov momentum, Adam, AdamW, RMSProp, etc and outperforms in all cases. Full paper here:
```
https://arxiv.org/pdf/2412.18052
```

## Features

The package provides a number of features for the example implementation of GAF. Specifically, it:

- Implements Gradient Agreement Filtering based on cosine distance. 
- Supports multiple optimizers: SGD, SGD with Nesterov momentum, Adam, AdamW, RMSProp.
- Provides examples on application of GAF in training image classifiers on CIFAR-100 and CIFAR-100N-Fine datasets.
   - Allows for label noise injection by flipping a percentage of labels.
   - Customizable hyperparameters via command-line arguments.
   - Logging and tracking with [Weights & Biases (wandb)](https://wandb.ai/).

## Installation

There are a few ways to install the package and run the examples.

**Local Installation**

To install the package and run the examples locally you can execute the following commands:

You can install via:
```bash
git clone https://github.com/Fchaubard/gradient_agreement_filtering.git
cd gradient_agreement_filtering
pip install .
```

If you wish to install with wandb, for the last step instead use

```bash
pip install ".[all]"
```

**Local Installation (Development)** If you wish to install the package with added
development dependencies, you can execute the following commands:

```bash
git clone https://github.com/Fchaubard/gradient_agreement_filtering.git
cd gradient_agreement_filtering
pip install ".[dev]"
```

**PyPI Installation**

If you only with to take advantage of the package and not the examples, you can install the package via PyPI:

```bash
pip install gradient-agreement-filtering
```
   

## Usage

We provide two ways to easily incorporate GAF into your existing training. 
1. `step_GAF()`:
   If you want to use GAF inside your existing train loop, you can just replace your typical:

   ```
   ...
   optimizer.zero_grad()
   outputs = model(batch)
   loss = criterion(outputs, labels)
   loss.backward()
   optimizer.step()
   ...
   ```
   
   with one call to step_GAF() as per below:
   
   ```
   from gradient_agreement_filtering import step_GAF
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
   
2. `train_GAF()`:

   If you want to use GAF as the train loop, you can just replace your typical hugging face / keras style interface:

   ```
   trainer.Train()
   ```
   
   with one call to train_GAF() as per below:
   
   ```
   from gradient_agreement_filtering import train_GAF
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
   
### NOTE: Running with wandb

If you want to run with wandb, you will need to set your WANDB_API_KEY. You can do this in a few ways:

1. You can login on the system first then run the .py via:
```bash
wandb login <your-wandb-api-key>
```

2. Add the following line to the top of your .py file:

```python
os.environ["WANDB_API_KEY"] = "<your-wandb-api-key>"
```

3. Or you can prepend any of the calls below with:

```bash
WANDB_API_KEY=<your-wandb-api-key> python *.py 
```

## Examples

We provide three examples to demonstrate the use of GAF in training image classifiers on CIFAR-100 and CIFAR-100N-Fine datasets.

### 1_cifar_100_train_loop_exposed.py

This file uses `step_GAF()` to train a ResNet18 model on the CIFAR-100 dataset using PyTorch with the ability to add noise to the labels to observe how GAF performs under noisy conditions. The code supports various optimizers and configurations, allowing you to experiment with different settings to understand the impact of GAF on model training.

Example call:
```bash
python examples/1_cifar_100_train_loop_exposed.py --GAF True --optimizer "SGD+Nesterov+val_plateau" --learning_rate 0.01 --momentum 0.9 --nesterov True --wandb True --verbose True --num_samples_per_class_per_batch 1 --num_batches_to_force_agreement 2 --label_error_percentage 0.15 --cos_distance_thresh 0.97
```

### 2_cifar_100_trainer.py
This file uses `train_GAF()` to train a ResNet18 model on the CIFAR-100 dataset using PyTorch just to show how it works. 

Example call:
```
python examples/2_cifar_100_trainer.py 
```

### 3_cifar_100N_train_loop_exposed.py

This file uses `step_GAF()` to train a ResNet34 model on the CIFAR-100N-Fine dataset using PyTorch to observe how GAF performs under typical labeling noise. The code supports various optimizers and configurations, allowing you to experiment with different settings to understand the impact of GAF on model training.

Example call:
```bash
python examples/3_cifar_100N_Fine_train_loop_exposed.py --GAF True --optimizer "SGD+Nesterov+val_plateau"  --cifarn True --learning_rate 0.01 --momentum 0.9 --nesterov True --wandb True --verbose True --num_samples_per_class_per_batch 2 --num_batches_to_force_agreement 2 --cos_distance_thresh 0.97
```

## Running sweeps

We also provide a shell script to run a sweep for convenience. The script spawns screen sessions and will randomly allocate the runs to GPUs. It can be run multiple times without . You should make sure you update the script with your own WANDB_API_KEY before running. Here is how you run it:

### test/run_sweeps.sh

Example call:
```bash
cd test
chmod +x run_sweeps.sh
./run_sweeps.sh
```


## Acknowledgement

To cite this work, please use the following BibTeX entry:

```
We would like to acknowledge and thank Alex Tzikas, Harrison Delecki, and Francois Chollet who provided invaluable help through discussions and feedback.
```

## License

This package is licensed under the MIT license. See [LICENSE](LICENSE) for details.

