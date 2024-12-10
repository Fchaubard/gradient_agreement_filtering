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

# Gradient Agreement Filtering (GAF) with ResNet18 on CIFAR-100

This repository provides an implementation of **Gradient Agreement Filtering (GAF)** applied to training a ResNet18 model on the CIFAR-100 dataset using PyTorch. The code supports various optimizers and configurations, allowing you to experiment with different settings to understand the impact of GAF on model training.

# Gradient Agreement Filtering (GAF) with ResNet34 on CIFAR-100N-Fine
```
TODO:
```

# Gradient Agreement Filtering (GAF) with ViT on ImageNet
```
TODO: 
```

# Gradient Agreement Filtering (GAF) with Pythia on GSM8k
```
TODO:  
```


# Gradient Agreement Filtering (GAF) with Pythia on GSM8k
```
TODO:  
```


## Requirements

- Python 3.6 or higher
- PyTorch 1.7 or higher
- torchvision 0.8 or higher
- numpy
- wandb

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/gaf-cifar100.git
   pip install .
   ```

## Usage

1. **step_GAF():**

   ```bash
   git clone https://github.com/your_username/gaf-cifar100.git
   pip install .
   ```

## Examples

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/gaf-cifar100.git
   pip install .
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

