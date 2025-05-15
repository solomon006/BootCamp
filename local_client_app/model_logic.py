# model_logic.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import glob  # Для удобного поиска файлов
from datetime import datetime # Перенесем импорт datetime сюда для get_current_time_str

# --- Глобальные настройки для модели ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["cat", "dog"]  # 0: cat, 1: dog
MODEL_SAVE_PATH = 'current_team_model.pth'
NUM_CLASSES = len(CLASS_NAMES)

# !!! ДОБАВЛЕННЫЕ СТРОКИ ДЛЯ ИСПРАВЛЕНИЯ ОШИБКИ !!!
TEAM_UPLOAD_BASE_PATH = 'team_uploads'        # Базовый путь для загрузок команды
COMMON_TEST_SET_PATH = 'common_test_set'      # Путь к общему тестовому набору
# !!! КОНЕЦ ДОБАВЛЕННЫХ СТРОК !!!

# --- Трансформации изображений ---
# Для обучающих изображений (загружаемых студентами), с аугментацией
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Небольшое изменение цвета
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Стандартная нормализация ImageNet
])

# Для тестовых изображений (из общего набора), без аугментации
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Определение и инициализация модели ---
criterion = nn.CrossEntropyLoss()
model = None
optimizer = None

def get_current_time_str(): # Определение функции до ее первого вызова в init_model
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def pretrain_on_existing_uploads(num_iterations_per_image=1):
    print(f"[{get_current_time_str()}] Запуск предварительного обучения на существующих загруженных изображениях...")
    images_processed = 0
    # Убедимся, что TEAM_UPLOAD_BASE_PATH существует
    if not os.path.isdir(TEAM_UPLOAD_BASE_PATH):
        print(f"[{get_current_time_str()}] Папка для загрузок команды '{TEAM_UPLOAD_BASE_PATH}' не найдена. Предварительное обучение пропускается.")
        return

    for label_int, class_name in enumerate(CLASS_NAMES):
        upload_folder = os.path.join(TEAM_UPLOAD_BASE_PATH, class_name)
        if not os.path.isdir(upload_folder):
            print(f"[{get_current_time_str()}] Предварительное обучение: Папка не найдена {upload_folder}, пропуск.")
            continue

        image_files = glob.glob(os.path.join(upload_folder, "*.jpg")) + \
                      glob.glob(os.path.join(upload_folder, "*.jpeg")) + \
                      glob.glob(os.path.join(upload_folder, "*.png"))

        if not image_files:
            print(f"[{get_current_time_str()}] Предварительное обучение: Изображения не найдены в {upload_folder} для класса {class_name}.")
            continue
        
        print(f"[{get_current_time_str()}] Предварительное обучение: Найдено {len(image_files)} изображений в {upload_folder} для класса {class_name}.")

        for image_path in image_files:
            # Проверяем, что это файл, а не папка (на всякий случай, хотя glob обычно возвращает файлы)
            if not os.path.isfile(image_path):
                continue
            print(f"[{get_current_time_str()}] Предварительное обучение на: {image_path}")
            # Вызываем train_on_new_image, но не сохраняем модель после каждого изображения
            success = train_on_new_image(image_path, label_int, num_iterations=num_iterations_per_image, save_model_after_this_image=False)
            if success:
                images_processed += 1
    
    if images_processed > 0:
        # Сохраняем модель один раз после обработки всех изображений в предобучении
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_SAVE_PATH)
            print(f"[{get_current_time_str()}] Модель сохранена после завершения предварительного обучения на {images_processed} изображениях.")
        except Exception as e:
            print(f"[{get_current_time_str()}] Ошибка сохранения модели после предварительного обучения: {e}")
    else:
        print(f"[{get_current_time_str()}] Предварительное обучение: Изображения для обработки не найдены или не обработаны.")

def init_model():
    global model, optimizer
    # Создаем SqueezeNet 1.1 БЕЗ предобученных весов (обучение с нуля)
    model = models.squeezenet1_1(weights=None)
    # Заменяем классификатор для нашего количества классов
    model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1))
    model.num_classes = NUM_CLASSES
    model = model.to(DEVICE)

    # Оптимизатор (Adam хорошо подходит для начала)
    # lr=0.001 обычно неплох для обучения с нуля, но для инкрементального может быть много
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Уменьшаем learning rate

    # Пытаемся загрузить сохраненное состояние, если оно есть
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"[{get_current_time_str()}] Модель и состояние оптимизатора успешно загружены из {MODEL_SAVE_PATH}")
        except Exception as e:
            print(
                f"[{get_current_time_str()}] Ошибка загрузки модели из {MODEL_SAVE_PATH}: {e}. Инициализируем оптимизатор заново.")
            # Если ошибка загрузки состояния оптимизатора (например, из-за изменения модели),
            # переинициализируем оптимизатор для текущей структуры модели
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Уменьшаем learning rate и здесь
    else:
        print(f"[{get_current_time_str()}] Файл {MODEL_SAVE_PATH} не найден. Инициализирована новая модель.")
        # Вызов функции предварительного обучения, если модель новая
        print(f"[{get_current_time_str()}] Запуск начального обучения на существующих загруженных изображениях, так как сохраненная модель не найдена.")
        pretrain_on_existing_uploads(num_iterations_per_image=2) # Используем 2 итерации для каждого изображения при предобучении
        # train_on_new_image внутри pretrain_on_existing_uploads сохранит модель после последнего изображения


# --- Функция инкрементального обучения на новом изображении ---
def train_on_new_image(image_path_str: str, label_int: int, num_iterations: int = 5, save_model_after_this_image: bool = True):
    if model is None or optimizer is None:
        # Эта проверка теперь менее вероятна, так как init_model() вызывается при импорте
        print(f"[{get_current_time_str()}] Модель или оптимизатор не инициализированы.")
        init_model() # Попытка инициализировать, если вдруг не были
        if model is None or optimizer is None: # Если все еще нет, выходим
             return False


    model.train()  # Переводим модель в режим обучения
    try:
        img = Image.open(image_path_str).convert("RGB")
        img_tensor = train_transform(img)  # Применяем аугментацию и трансформацию
        img_tensor = img_tensor.to(DEVICE).unsqueeze(0)  # Отправляем на DEVICE и добавляем батч-измерение
        label_tensor = torch.tensor([label_int], device=DEVICE, dtype=torch.long)

        current_loss = 0.0
        for i in range(num_iterations):
            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(img_tensor)  # Прямой проход
            loss = criterion(outputs, label_tensor)  # Считаем потери
            loss.backward()  # Обратный проход (вычисление градиентов)
            optimizer.step()  # Шаг оптимизатора (обновление весов)
            current_loss = loss.item()  # Запоминаем текущие потери

        print(
            f"[{get_current_time_str()}] Модель дообучена на {os.path.basename(image_path_str)} ({num_iterations} итераций), последняя потеря: {current_loss:.4f}")

        # Сохраняем модель и состояние оптимизатора, только если указано
        if save_model_after_this_image:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_SAVE_PATH)
            print(f"[{get_current_time_str()}] Модель сохранена после обучения на {os.path.basename(image_path_str)}")
        return True
    except Exception as e:
        print(f"[{get_current_time_str()}] Ошибка во время инкрементального обучения на {image_path_str}: {e}")
        return False


# --- Функция загрузки изображений для оценки ---
def load_images_for_evaluation(base_test_folder_path: str):
    images = []
    labels = []
    print(f"[{get_current_time_str()}] Загрузка тестовых изображений из: {base_test_folder_path}")
    for i, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(base_test_folder_path, class_name)
        if not os.path.exists(folder):
            print(f"[{get_current_time_str()}] ПРЕДУПРЕЖДЕНИЕ: Папка для теста не найдена: {folder}")
            continue
        # Используем glob для поиска всех .jpg, .jpeg, .png файлов
        for image_file in glob.glob(os.path.join(folder, "*.jpg")) + \
                          glob.glob(os.path.join(folder, "*.jpeg")) + \
                          glob.glob(os.path.join(folder, "*.png")):
            try:
                img = Image.open(image_file).convert("RGB")
                img_tensor = test_transform(img)  # Используем test_transform (без аугментации)
                images.append(img_tensor)
                labels.append(i)  # 0 для cat, 1 для dog
            except Exception as e:
                print(f"[{get_current_time_str()}] Ошибка загрузки тестового изображения {image_file}: {e}")
    if not images:
        print(f"[{get_current_time_str()}] ПРЕДУПРЕЖДЕНИЕ: Тестовые изображения не найдены или не удалось загрузить.")
    return images, labels


# --- Функция оценки на общем тестовом наборе ---
def evaluate_on_common_test_set(common_test_images_tensors, common_test_labels_list):
    if model is None:
        print(f"[{get_current_time_str()}] Модель не инициализирована для оценки.")
        return 0.0
    if not common_test_images_tensors:
        print(f"[{get_current_time_str()}] Нет данных для оценки.")
        return 0.0

    model.eval()  # Переводим модель в режим оценки
    correct = 0
    total = 0

    with torch.no_grad():  # Отключаем вычисление градиентов для оценки
        for i in range(len(common_test_images_tensors)):
            img_tensor = common_test_images_tensors[i].to(DEVICE).unsqueeze(0)
            label_int = common_test_labels_list[i]
            label_tensor = torch.tensor([label_int], device=DEVICE, dtype=torch.long)

            outputs = model(img_tensor)
            _, predicted = torch.max(outputs.data, 1)  # Получаем индекс класса с максимальной вероятностью

            total += 1
            if predicted == label_tensor:
                correct += 1

    accuracy = 100 * correct / total if total > 0 else 0.0
    print(f"[{get_current_time_str()}] Точность на общем тестовом наборе: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


# Инициализируем модель при импорте модуля
init_model()