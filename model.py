import torch
from torchvision import models
def get_vgg16(num_classes):
    # Загружаем предобученную VGG16
    model = models.vgg16(weights="IMAGENET1K_V1")

    # Замораживаем все слои, кроме классификатора
    for param in model.features.parameters():
        param.requires_grad = False

    # Настраиваем классификатор
    model.classifier[-1] = torch.nn.Linear(4096, num_classes)  # Последний слой классификатора
    return model