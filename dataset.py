from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(data_dir, batch_size):
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # Заменяем путь для train и validation на вашу структуру данных
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=data_transforms["train"])
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/validation", transform=data_transforms["val"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader