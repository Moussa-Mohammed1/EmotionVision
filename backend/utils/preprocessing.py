import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transform():
    return transforms.Compose((
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ))

transform = get_transform()

train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root="dataset/test",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

images, labels = next(iter(train_loader))

