import torch
import torch.optim as optim
import os
import glob
from model import get_vgg16
from dataset import create_dataloaders
from utils import train_one_epoch, validate_one_epoch


# Параметры обучения
data_dir = "dataset"  # Путь к вашей папке с датасетом
num_classes = 23  # Количество классов (если известно)
num_epochs = 6
batch_size = 32
lr = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создание загрузчиков данных
train_loader, val_loader = create_dataloaders(data_dir, batch_size)

# Инициализация модели
model = get_vgg16(num_classes)
model = model.to(device)

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 10)

    # Обучение
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    # Валидация
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Сохранение чекпоинта только на нечетных эпохах
    if (epoch + 1) % 2 == 1:
        checkpoint_name = f"model_epoch_{epoch + 1}_valacc_{val_acc:.2f}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            'epoch': epoch,
        }, 'checkpoint.pth')
        print(f"Model checkpoint saved: {checkpoint_name}")

        # Удаление старых чекпоинтов, если их больше 10
        checkpoints = sorted(glob.glob("model_epoch_*.pth"))
        if len(checkpoints) > 10:
            os.remove(checkpoints[0])
            print(f"Removed old checkpoint: {checkpoints[0]}")