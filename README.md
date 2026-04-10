# RAFed

RAFed is a federated learning framework for responsive data augmentation.  
This repository contains a runnable implementation of the paper-oriented training loop, including:

- shared augmentation policy learning across clients
- local model updates with augmented samples
- periodic policy updates to reduce communication and computation cost
- collaborative server aggregation for both model and policy parameters

## Repository Layout

- `run_rafed.py`: recommended entry point
- `RAFed_Col.py`: collaborative optimization runner
- `RAFed.py`: legacy runner kept for compatibility
- `utils/`: compatibility package that mirrors the original flat module layout
- `augment_func.py`, `augment_utils.py`: augmentation operators and policy sampling
- `train_utils.py`: dataset split, loaders, and helper utilities
- `Faal_step.py`, `Faal_step_col.py`: client/server optimization logic

## Main Configuration

The default configuration in `RAFed_Col.py` is aligned with the method description:

- `policy_update_interval = 10`
- `H1 = 25`, `H2 = 25`
- `w_inner_lr = 0.2`
- `p_inner_lr = 0.4`
- `gradclip_policy = 0.45`
- `eps = 0.1`
- `policy_lr = 0.7`

The augmentation policy operates on the 17-operation space described in the paper and samples two sequential operations from a `17 x 17` joint distribution.

## Requirements

The code expects a Python environment with:

- PyTorch
- torchvision
- numpy
- pillow
- tqdm
- scipy / scikit-learn where applicable
- tensorboard or tensorboardX

The exact package versions depend on the PyTorch release you use.  
The code also uses CUDA tensors in the training loop, so a GPU-enabled environment is strongly recommended.

## Data

The datasets are downloaded automatically into the `data/` folder on first run.

Supported datasets in the provided scripts:

- CIFAR-10
- CIFAR-100
- SVHN
- Tiny-ImageNet

## Quick Start

Run the collaborative version:

```bash
python run_rafed.py --dataset cifar10 --num_users 100 --ood_users 30 --alpha 0.1
```

If you want to invoke the original script directly:

```bash
python RAFed_Col.py --dataset cifar10 --num_users 100 --ood_users 30 --alpha 0.1
```

## Notes

- `RAFed_Col.py` is the recommended entry point for the collaborative optimization setup.
- `RAFed.py` is preserved as a legacy variant.
- Training logs are written under `runs/`.
- The `utils/` directory is a compatibility shim so that the legacy `from utils.* import ...` imports keep working without rewriting every source file.

