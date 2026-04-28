import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# global settings
DATA_DIR = './data'
BATCH_SIZE = 32
IMAGE_SIZE = 128 # scaled down to 128x128 so my laptop doesn't catch fire

def get_dataloaders(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
    """
    Grabs the oxford pets dataset, sorts out the train/val split and wraps them in loaders.
    """
    
    # 1. setup the transforms
    # using standard imagenet numbers for normalization, seems to work well enough
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 2. pull the actual data
    # this gets both train and val together initially
    full_train_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, 
        split='trainval', 
        target_types='category', 
        download=True, 
        transform=train_transforms
    )

    # carving out about 10% for validation so I can see if it's overfitting
    val_size = int(0.1 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    
    # locking the seed down so the split is always the same
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # this part is annoying - random_split keeps the train augments for the val set too.
    # if I just override it, it ruins the train set.
    val_dataset.dataset.transform = test_val_transforms # wait, this breaks full_train_dataset!
    
    # okay, proper fix: just reload the dataset completely fresh for validation
    # so we don't mix up the transforms.
    val_dataset_clean = datasets.OxfordIIITPet(
        root=DATA_DIR, 
        split='trainval', 
        target_types='category', 
        download=False, 
        transform=test_val_transforms
    )
    # use the indices from before to grab the right images
    val_dataset = torch.utils.data.Subset(val_dataset_clean, val_dataset.indices)
    
    # re-loading the train dataset too just to be absolutely certain it's clean
    train_dataset_clean = datasets.OxfordIIITPet(
        root=DATA_DIR, 
        split='trainval', 
        target_types='category', 
        download=False, 
        transform=train_transforms
    )
    train_dataset = torch.utils.data.Subset(train_dataset_clean, train_dataset.indices)

    # grab the final test set
    test_dataset = datasets.OxfordIIITPet(
        root=DATA_DIR, 
        split='test', 
        target_types='category', 
        download=True, 
        transform=test_val_transforms
    )

    # 3. wrap everything in dataloaders and we're good to go
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )

    return train_loader, val_loader, test_loader, len(train_dataset), len(val_dataset), len(test_dataset)

if __name__ == '__main__':
    train_loader, val_loader, test_loader, num_train, num_val, num_test = get_dataloaders()
    print(f"Train size: {num_train}")
    print(f"Val size: {num_val}")
    print(f"Test size: {num_test}")
    
    # Get a batch to check shapes
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Number of classes (from max label): {labels.max().item() + 1}")
