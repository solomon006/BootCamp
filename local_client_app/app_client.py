# app_client.py
from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename  # Для безопасного сохранения имен файлов
import model_logic  # Наш модуль с логикой модели
import time
import requests  # Для отправки на сервер

# import schedule # Закомментировано для упрощения, см. примечание ниже
# import threading

# --- Настройки Flask ---
app = Flask(__name__)
app.secret_key = "super_secret_key_for_event"  # Установите секретный ключ для flash-сообщений

# Папки для загружаемых студентами изображений
# Убедитесь, что эти пути корректны относительно app_client.py
# или используйте абсолютные пути
TEAM_NAME = "MyAwesomeTeam"  # ЗАМЕНИТЬ: Уникальное имя/ID для каждой команды
UPLOAD_FOLDER_CATS = os.path.join(model_logic.TEAM_UPLOAD_BASE_PATH, 'cat')
UPLOAD_FOLDER_DOGS = os.path.join(model_logic.TEAM_UPLOAD_BASE_PATH, 'dog')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER_CATS'] = UPLOAD_FOLDER_CATS
app.config['UPLOAD_FOLDER_DOGS'] = UPLOAD_FOLDER_DOGS

# Создаем папки, если их нет
os.makedirs(UPLOAD_FOLDER_CATS, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_DOGS, exist_ok=True)

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
                           num_dogs=num_dogs)


@app.route('/upload', methods=['POST'])
def upload_file():
    global current_local_accuracy, last_accuracy_update_time
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
        label_int = -1

        if label_str == 'cat':
            save_path = os.path.join(app.config['UPLOAD_FOLDER_CATS'], filename_with_ts)
            label_int = 0
        elif label_str == 'dog':
            save_path = os.path.join(app.config['UPLOAD_FOLDER_DOGS'], filename_with_ts)
            label_int = 1
        else:
            flash('Неверная метка класса')
            return redirect(request.url)

        try:
            file.save(save_path)
            flash(f'Фото "{filename}" успешно загружено как "{label_str}". Начинаем дообучение...')
            print(f"[{model_logic.get_current_time_str()}] Файл сохранен: {save_path}")

            # Дообучаем модель на новом изображении
            training_success = model_logic.train_on_new_image(save_path, label_int,
                                                              num_iterations=3)  # Уменьшаем количество итераций

            if training_success:
                flash('Модель дообучена на новом изображении!')
                # Сразу после успешного обучения проводим оценку и отправляем на сервер
                # (упрощенный вариант без отдельного планировщика)
                print(f"[{model_logic.get_current_time_str()}] Запуск оценки после загрузки...")
                current_local_accuracy = model_logic.evaluate_on_common_test_set(COMMON_TEST_IMAGES, COMMON_TEST_LABELS)
                last_accuracy_update_time = model_logic.get_current_time_str()
                send_score_to_server(TEAM_NAME, current_local_accuracy)

            else:
                flash('Ошибка во время дообучения модели.')

        except Exception as e:
            flash(f'Произошла ошибка при сохранении или обучении: {e}')
            print(f"[{model_logic.get_current_time_str()}] Ошибка при сохранении/обучении: {e}")

        return redirect(url_for('index'))
    else:
        flash('Недопустимый тип файла или не выбрана метка.')
        return redirect(request.url)


# --- Связь с центральным сервером ---
CENTRAL_SERVER_URL_SUBMIT = "http://127.0.0.1:5000/submit_score"  # ЗАМЕНИТЬ: IP и порт вашего центрального сервера


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


# --- Запуск Flask приложения ---
if __name__ == '__main__':
    print(f"[{model_logic.get_current_time_str()}] Запуск Flask-приложения для команды: {TEAM_NAME}")
    print(
        f"[{model_logic.get_current_time_str()}] Общий тестовый набор содержит {len(COMMON_TEST_IMAGES)} изображений.")
    print(f"[{model_logic.get_current_time_str()}] Устройство для PyTorch: {model_logic.DEVICE}")
    # Для доступа по сети используйте host='0.0.0.0'
    # debug=True НЕ ИСПОЛЬЗОВАТЬ В ПРОДАШЕНЕ/НА МЕРОПРИЯТИИ из-за безопасности и производительности
    # Для мероприятия лучше использовать debug=False
    app.run(host='0.0.0.0', port=8080, debug=False)