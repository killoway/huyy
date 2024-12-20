import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import get_vgg16
from utils import train_one_epoch, validate_one_epoch
import os


class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("800x600")

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Интеграция кнопок и элементов интерфейса
        self.load_image_button = tk.Button(root, text="Загрузить изображение", command=self.load_image)
        self.load_image_button.pack(pady=10)

        self.classify_button = tk.Button(root, text="Классифицировать", command=self.classify_image, state=tk.DISABLED)
        self.classify_button.pack(pady=10)

        self.result_label = tk.Label(root, text="Результат: ", font=("Helvetica", 14))
        self.result_label.pack(pady=20)

        self.probability_label = tk.Label(root, text="Вероятность: ", font=("Helvetica", 12))
        self.probability_label.pack(pady=10)

        self.model_info_label = tk.Label(root, text="Модель не загружена", font=("Helvetica", 12))
        self.model_info_label.pack(pady=10)

        self.select_model_button = tk.Button(root, text="Выбрать модель или чекпоинт", command=self.select_model)
        self.select_model_button.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Выберите изображение",
                                               filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            image = image.resize((224, 224))  # Resize for VGG16 input
            self.display_image(image)
            self.classify_button.config(state=tk.NORMAL)

    def display_image(self, image):
        # Если уже есть изображение, удаляем его
        if hasattr(self, 'image_label') and self.image_label.winfo_exists():
            self.image_label.destroy()

        # Отображаем новое изображение
        self.img = ImageTk.PhotoImage(image)
        self.image_label = tk.Label(self.root, image=self.img)
        self.image_label.pack(pady=20)

    def classify_image(self):
        if not hasattr(self, 'model'):
            messagebox.showerror("Ошибка", "Модель не загружена")
            return

        image = Image.open(self.image_path).convert('RGB')  # Убедитесь, что изображение в RGB
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = preprocess(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            _, pred = torch.max(output, 1)

        class_idx = pred.item()

        # Список названий классов
        class_names = [
            "Балетки", "Ботинки", "Челси", "Сабо", "Эспандрильи", "Валенки", "Рыбацкие сапоги", "Шлепанцы",
            "Галоши", "Ботфорты", "Лоферы", "Туфли", "Мокасины", "Оксфорды", "Сандалии",
            "Туфли на каблуке", "Тапочки", "Кроссовки", "Сникеры", "Кеды",
            "Треккинговая обувь", "Угги", "Женские сапоги"
        ]

        if class_idx < len(class_names):
            predicted_class = class_names[class_idx]
        else:
            predicted_class = "Неизвестный класс"

        self.result_label.config(text=f"Результат: {predicted_class}")

        # Вывод вероятностей только для 5 самых больших классов
        self.show_top_5_class_probabilities(probabilities.cpu(), class_names)

    def show_top_5_class_probabilities(self, probabilities, class_names):
        # Сортируем вероятности, чтобы выбрать топ-5
        top5_probabilities, top5_indices = torch.topk(probabilities, 5)

        # Сформируем строку с вероятностями для топ-5 классов
        probability_text = "Топ-5 вероятностей по классам:\n"
        for i in range(5):
            class_idx = top5_indices[i].item()
            probability_text += f"{class_names[class_idx]}: {top5_probabilities[i]:.4f}\n"

        # Отобразим текст с вероятностями
        if hasattr(self, 'probability_label'):
            self.probability_label.config(text=probability_text)
        else:
            self.probability_label = tk.Label(self.root, text=probability_text, font=("Helvetica", 10))
            self.probability_label.pack(pady=10)
    def select_model(self):
        file_path = filedialog.askopenfilename(title="Выберите модель или чекпоинт",
                                               filetypes=[("Model Files", "*.pth")])
        if file_path:
            self.load_model(file_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)

        model = get_vgg16(num_classes=23)  # Укажите количество классов в модели
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model.to(self.device)

        # Извлекаем данные из checkpoint
        epoch = checkpoint.get('epoch', 'Неизвестно')
        train_loss = checkpoint.get('train_loss', 'Неизвестно')
        train_accuracy = checkpoint.get('train_accuracy', 'Неизвестно')
        val_loss = checkpoint.get('val_loss', 'Неизвестно')
        val_accuracy = checkpoint.get('val_accuracy', 'Неизвестно')

        # Обновляем информацию о модели
        model_info_text = (
            f"Модель загружена: {os.path.basename(model_path)}\n"
            f"Эпоха: {epoch}\n"
            f"Обучение - Потери: {train_loss:.4f}, Точность: {train_accuracy:.4f}\n"
            f"Валидация - Потери: {val_loss:.4f}, Точность: {val_accuracy:.4f}"
        )

        self.model_info_label.config(text=model_info_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
