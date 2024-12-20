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

        self.model_info_label = tk.Label(root, text="Модель не загружена", font=("Helvetica", 12))
        self.model_info_label.pack(pady=10)

        self.select_model_button = tk.Button(root, text="Выбрать модель", command=self.select_model)
        self.select_model_button.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Выберите изображение", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            image = image.resize((224, 224))  # Resize for VGG16 input
            self.display_image(image)
            self.classify_button.config(state=tk.NORMAL)

    def display_image(self, image):
        self.img = ImageTk.PhotoImage(image)
        self.image_label = tk.Label(self.root, image=self.img)
        self.image_label.pack(pady=20)

    def classify_image(self):
        if not hasattr(self, 'model'):
            messagebox.showerror("Ошибка", "Модель не загружена")
            return

        image = Image.open(self.image_path)
        image = image.resize((224, 224))
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            _, pred = torch.max(output, 1)

        class_idx = pred.item()
        # Предполагаем, что у вас есть список классов, например, class_names
        class_names = [str(i) for i in range(23)]  # Пример для 23 классов
        predicted_class = class_names[class_idx]

        self.result_label.config(text=f"Результат: {predicted_class}")

    def select_model(self):
        file_path = filedialog.askopenfilename(title="Выберите модель", filetypes=[("Model Files", "*.pth")])
        if file_path:
            self.load_model(file_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = get_vgg16(num_classes=23)  # Укажите количество классов в модели
        model.load_state_dict(checkpoint['model_state_dict'])
        self.model = model.to(self.device)

        self.model_info_label.config(text=f"Модель загружена: {os.path.basename(model_path)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
