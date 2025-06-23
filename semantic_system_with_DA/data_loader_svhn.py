import torch
from torchvision import datasets  # Provides access to standard datasets like MNIST, SVHN
from torchvision import transforms  # Provides image transformation utilities

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN datasets.
    
    Args:
        config: Configuration object containing parameters like:
            - image_size: Target size for image resizing
            - batch_size: Number of samples per batch
            - num_workers: Number of subprocesses for data loading
            - svhn_path: Path to SVHN dataset
            - mnist_path: Path to MNIST dataset
    
    Returns:
        Tuple of (svhn_loader, mnist_loader) PyTorch DataLoader objects
    """

    # Define image transformations for SVHN dataset:
    # 1. Scale to target image size
    # 2. Convert to PyTorch Tensor
    # 3. Normalize with mean=0.5 and std=0.5 for all 3 channels (RGB)
    transform = transforms.Compose([
        transforms.Resize(config.image_size),  # Resize image
        transforms.ToTensor(),  # Convert to tensor (values 0-1)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1,1] range
    ])

    # Define separate transformations for MNIST dataset:
    # MNIST is grayscale so normalization only needs single value
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor first (values 0-1)
        transforms.Resize(config.image_size),  # Then resize
        transforms.Normalize(mean=0.5, std=0.5)  # Normalize to [-1,1] range
    ])

    # Load SVHN dataset (Street View House Numbers)
    # - root: Directory where dataset is/will be stored
    # - download: If True, downloads the dataset if not available
    # - transform: Applies the defined transformations
    svhn = datasets.SVHN(
        root=config.svhn_path,
        download=True,
        transform=transform
    )

    # Load MNIST dataset (Handwritten digits)
    # Note: Original commented line used the same transform as SVHN (incorrect for grayscale)
    # Correct version uses transform_mnist specifically designed for MNIST
    mnist = datasets.MNIST(
        root=config.mnist_path,
        download=True,
        transform=transform_mnist
    )

    # Create PyTorch DataLoader for SVHN:
    # - dataset: SVHN dataset object
    # - batch_size: Number of samples per batch
    # - shuffle: If True, shuffles the data every epoch
    # - num_workers: Number of subprocesses for data loading
    svhn_loader = torch.utils.data.DataLoader(
        dataset=svhn,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    # Create PyTorch DataLoader for MNIST with same parameters
    mnist_loader = torch.utils.data.DataLoader(
        dataset=mnist,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    return svhn_loader, mnist_loader


"""
Key points about this implementation:

Dataset Differences:

SVHN is a 3-channel (RGB) dataset of real-world house numbers

MNIST is a single-channel (grayscale) dataset of handwritten digits

Transformations:

Separate transformation pipelines handle the different image types correctly

Both pipelines:

Resize images to the target size

Convert to PyTorch tensors

Normalize pixel values to [-1, 1] range

Data Loading:

Uses PyTorch's DataLoader for efficient batching and shuffling

Supports multi-process data loading through num_workers

Automatically downloads datasets if not present

Important Note:

The original code had MNIST using the same transform as SVHN (commented out)

The current version correctly uses separate transforms accounting for:

MNIST being single-channel vs SVHN's 3 channels

Different normalization requirements
"""