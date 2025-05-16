# app_client.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename  # Для безопасного сохранения имен файлов
import model_logic  # Наш модуль с логикой модели
import time
import requests  # Для отправки на сервер
import threading
import glob
import atexit
import torch

# --- Настройки Flask ---
app = Flask(__name__)
app.secret_key = "super_secret_key_for_event"  # Установите секретный ключ для flash-сообщений

# Флаг для отслеживания новых загруженных изображений
NEW_IMAGES_ADDED = False
# Событие для остановки фоновых потоков
stop_background_threads = threading.Event()
# Блокировка для безопасной работы с моделью из разных потоков
model_lock = threading.Lock()

# Папки для загружаемых студентами изображений
# Убедитесь, что эти пути корректны относительно app_client.py
# или используйте абсолютные пути
TEAM_NAME = "school 4"  # ЗАМЕНИТЬ: Уникальное имя/ID для каждой команды
UPLOAD_FOLDER_CATS = os.path.join(model_logic.TEAM_UPLOAD_BASE_PATH, 'cat')
UPLOAD_FOLDER_DOGS = os.path.join(model_logic.TEAM_UPLOAD_BASE_PATH, 'dog')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER_CATS'] = UPLOAD_FOLDER_CATS
app.config['UPLOAD_FOLDER_DOGS'] = UPLOAD_FOLDER_DOGS

# Создаем папки, если их нет
os.makedirs(UPLOAD_FOLDER_CATS, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_DOGS, exist_ok=True)

# URL для получения всех очков с центрального сервера
CENTRAL_SERVER_LEADERBOARD_URL = "https://34ca-46-251-217-124.ngrok-free.app/leaderboard_data" # ИЗМЕНЕНО: на актуальный эндпоинт сервера

# Глобальная переменная для хранения текущей точности (для упрощения)
current_local_accuracy = 0.0
last_accuracy_update_time = "Еще не оценивалась"

# Загружаем общий тестовый набор один раз при старте
COMMON_TEST_IMAGES, COMMON_TEST_LABELS = model_logic.load_images_for_evaluation(model_logic.COMMON_TEST_SET_PATH)
if not COMMON_TEST_IMAGES:
    print(
        f"[{model_logic.get_current_time_str()}] КРИТИЧЕСКАЯ ОШИБКА: Общий тестовый набор не загружен. Оценка не будет работать.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# --- Маршруты Flask ---
@app.route('/', methods=['GET'])
def index():
    # Получаем количество загруженных файлов для статистики
    num_cats = len(os.listdir(app.config['UPLOAD_FOLDER_CATS']))
    num_dogs = len(os.listdir(app.config['UPLOAD_FOLDER_DOGS']))
    return render_template('index.html',
                           accuracy=current_local_accuracy,
                           last_update_time=last_accuracy_update_time,
                           num_cats=num_cats,
                           num_dogs=num_dogs,
                           team_name=TEAM_NAME) # Добавим имя команды в главный шаблон


@app.route('/upload', methods=['POST'])
def upload_file():
    global current_local_accuracy, last_accuracy_update_time, NEW_IMAGES_ADDED
    if 'photo' not in request.files:
        flash('Файл не выбран')
        return redirect(request.url)
    file = request.files['photo']
    label_str = request.form.get('label')

    if file.filename == '':
        flash('Файл не выбран')
        return redirect(request.url)

    if file and allowed_file(file.filename) and label_str:
        filename = secure_filename(file.filename)
        # Добавляем временную метку к имени файла, чтобы избежать перезаписи
        timestamp = str(int(time.time()))
        filename_with_ts = f"{timestamp}_{filename}"

        save_path = ""
        
        if label_str == 'cat':
            save_path = os.path.join(app.config['UPLOAD_FOLDER_CATS'], filename_with_ts)
        elif label_str == 'dog':
            save_path = os.path.join(app.config['UPLOAD_FOLDER_DOGS'], filename_with_ts)
        else:
            flash('Неверная метка класса')
            return redirect(request.url)

        try:
            # Только сохраняем файл, но не обучаем модель сразу
            file.save(save_path)
            
            # Устанавливаем флаг, что у нас есть новые изображения для обучения
            NEW_IMAGES_ADDED = True
            
            flash(f'Фото "{filename}" успешно загружено как "{label_str}". Модель будет дообучена в фоновом режиме.')
            print(f"[{model_logic.get_current_time_str()}] Файл сохранен: {save_path}")

        except Exception as e:
            flash(f'Произошла ошибка при сохранении файла: {e}')
            print(f"[{model_logic.get_current_time_str()}] Ошибка при сохранении: {e}")

        return redirect(url_for('index'))
    else:
        flash('Недопустимый тип файла или не выбрана метка.')
        return redirect(request.url)


# --- Связь с центральным сервером ---
CENTRAL_SERVER_URL_SUBMIT = "https://34ca-46-251-217-124.ngrok-free.app/submit_score"  # ЗАМЕНИТЬ: IP и порт вашего центрального сервера


def send_score_to_server(team_id_str, accuracy_float):
    payload = {'team_id': team_id_str, 'accuracy': accuracy_float}
    try:
        response = requests.post(CENTRAL_SERVER_URL_SUBMIT, json=payload, timeout=10)  # Таймаут 10 секунд
        if response.status_code == 200:
            print(
                f"[{model_logic.get_current_time_str()}] Результат ({accuracy_float:.2f}%) для {team_id_str} успешно отправлен на сервер.")
            flash(f"Результат ({accuracy_float:.2f}%) отправлен на сервер!")
        else:
            error_msg = f"Ошибка отправки на сервер: {response.status_code} - {response.text}"
            print(f"[{model_logic.get_current_time_str()}] {error_msg}")
            flash(error_msg)
    except requests.exceptions.RequestException as e:  # Ловим ошибки сети/соединения
        error_msg = f"Критическая ошибка отправки на сервер (возможно, сервер недоступен): {e}"
        print(f"[{model_logic.get_current_time_str()}] {error_msg}")
        flash(error_msg)


@app.route('/leaderboard')
def leaderboard_page():
    """Отображает страницу с рейтингом."""
    return render_template('leaderboard.html', team_name=TEAM_NAME)

@app.route('/api/leaderboard_data')
def api_leaderboard_data():
    """Возвращает данные для таблицы рейтинга в формате JSON."""
    try:
        response = requests.get(CENTRAL_SERVER_LEADERBOARD_URL, timeout=5)
        if response.status_code == 200:
            scores = response.json()
            # Сортируем по убыванию точности
            sorted_scores = sorted(scores, key=lambda x: x.get('accuracy', 0), reverse=True)
            return jsonify(sorted_scores)
        else:
            print(f"[{model_logic.get_current_time_str()}] Ошибка получения рейтинга с сервера: {response.status_code} - {response.text}")
            return jsonify({"error": "Could not fetch leaderboard data from server", "status_code": response.status_code}), 500
    except requests.exceptions.RequestException as e:
        print(f"[{model_logic.get_current_time_str()}] Критическая ошибка получения рейтинга с сервера: {e}")
        return jsonify({"error": f"Could not connect to leaderboard server: {e}"}), 500


# Функция фонового обучения, запускается в отдельном потоке
def background_training_task():
    """Фоновая задача, которая запускается каждую минуту и обучает модель на всех новых изображениях"""
    global NEW_IMAGES_ADDED, current_local_accuracy, last_accuracy_update_time
    
    print(f"[{model_logic.get_current_time_str()}] Запущен фоновый поток обучения модели")
    
    while not stop_background_threads.is_set():
        if NEW_IMAGES_ADDED:
            with model_lock:  # Блокируем доступ к модели на время обучения
                print(f"[{model_logic.get_current_time_str()}] Запуск периодического дообучения...")
                
                processed_images = []  # Список обработанных изображений
                
                # Обучение на всех файлах в папках cat и dog
                for label_int, class_name in enumerate(model_logic.CLASS_NAMES):
                    upload_folder = os.path.join(model_logic.TEAM_UPLOAD_BASE_PATH, class_name)
                    if not os.path.isdir(upload_folder):
                        continue
                    
                    # Находим все файлы изображений
                    image_files = glob.glob(os.path.join(upload_folder, "*.jpg")) + \
                                  glob.glob(os.path.join(upload_folder, "*.jpeg")) + \
                                  glob.glob(os.path.join(upload_folder, "*.png"))
                    
                    if not image_files:
                        print(f"[{model_logic.get_current_time_str()}] Нет изображений для класса {class_name}")
                        continue
                    
                    print(f"[{model_logic.get_current_time_str()}] Найдено {len(image_files)} изображений для класса {class_name}")
                    
                    # Дообучаем модель на всех изображениях этого класса
                    # Но НЕ сохраняем модель после каждого изображения
                    for image_path in image_files:
                        if not os.path.isfile(image_path):
                            continue
                        
                        try:
                            print(f"[{model_logic.get_current_time_str()}] Дообучение на: {os.path.basename(image_path)}")
                            model_logic.train_on_new_image(
                                image_path, 
                                label_int, 
                                num_iterations=2,  # Уменьшаем количество итераций для ускорения
                                save_model_after_this_image=False  # Не сохраняем после каждого изображения
                            )
                            processed_images.append(image_path)  # Добавляем путь к обработанному изображению
                        except Exception as e:
                            print(f"[{model_logic.get_current_time_str()}] Ошибка при обучении на {image_path}: {e}")
                
                # После обработки всех изображений, сохраняем модель один раз
                if processed_images:
                    try:
                        print(f"[{model_logic.get_current_time_str()}] Сохранение модели после обработки {len(processed_images)} изображений...")
                        torch.save({
                            'model_state_dict': model_logic.model.state_dict(),
                            'optimizer_state_dict': model_logic.optimizer.state_dict(),
                        }, model_logic.MODEL_SAVE_PATH)
                        
                        # Оцениваем модель и отправляем результат на сервер
                        print(f"[{model_logic.get_current_time_str()}] Оценка модели на тестовом наборе...")
                        current_local_accuracy = model_logic.evaluate_on_common_test_set(COMMON_TEST_IMAGES, COMMON_TEST_LABELS)
                        last_accuracy_update_time = model_logic.get_current_time_str()
                        
                        # Отправляем результат на сервер
                        send_score_to_server(TEAM_NAME, current_local_accuracy)
                        
                        # Сбрасываем флаг
                        NEW_IMAGES_ADDED = False
                        print(f"[{model_logic.get_current_time_str()}] Обучение завершено, точность: {current_local_accuracy:.4f}")
                    except Exception as e:
                        print(f"[{model_logic.get_current_time_str()}] Ошибка при сохранении/оценке модели: {e}")
        
        # Ждем 1 минуту перед следующей проверкой
        for _ in range(6):  # 6 * 10 = 60 секунд (1 минута)
            if stop_background_threads.is_set():
                break
            time.sleep(10)

# Функция очистки для корректного завершения потоков
def cleanup_background_threads():
    print(f"[{model_logic.get_current_time_str()}] Остановка фоновых потоков...")
    stop_background_threads.set()

# Регистрируем функцию очистки
atexit.register(cleanup_background_threads)

# --- Запуск Flask приложения ---
if __name__ == '__main__':
    print(f"[{model_logic.get_current_time_str()}] Запуск Flask-приложения для команды: {TEAM_NAME}")
    print(
        f"[{model_logic.get_current_time_str()}] Общий тестовый набор содержит {len(COMMON_TEST_IMAGES)} изображений.")
    print(f"[{model_logic.get_current_time_str()}] Устройство для PyTorch: {model_logic.DEVICE}")
    # Запуск фонового потока для обучения
    background_thread = threading.Thread(target=background_training_task, daemon=True)
    background_thread.start()
    # Для доступа по сети используйте host='0.0.0.0'
    # debug=True НЕ ИСПОЛЬЗОВАТЬ В ПРОДАШЕНЕ/НА МЕРОПРИЯТИИ из-за безопасности и производительности
    # Для мероприятия лучше использовать debug=False
    app.run(host='0.0.0.0', port=8080, debug=False)