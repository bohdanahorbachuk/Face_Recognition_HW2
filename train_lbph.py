import cv2
import numpy as np
import os
import sys

# --- НАЛАШТУВАННЯ ---
DATA_DIR = 'faces_db'
MODEL_FILE = "lbph_model.yml"

def train_recognizer(data_dir=DATA_DIR):
    """
    Сканує папки користувачів, збирає зображення та навчає модель LBPH.
    """
    # Створення об'єкта розпізнавання обличчя LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    faces = []
    labels = []
    names = {}  # Словник для зберігання ID та імен
    label_id = 0

    print(f"Початок сканування бази даних облич у {data_dir}...")

    # Перевіряємо, чи існує папка бази даних
    if not os.path.isdir(data_dir):
        print(f"Помилка: Папка '{data_dir}' не знайдена. Спочатку зберіть фотографії!")
        sys.exit()

    for user_name in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_name)
        
        # Пропускаємо системні файли (.DS_Store) та не-папки
        if not os.path.isdir(user_path) or user_name.startswith('.'):
            continue
            
        names[label_id] = user_name
        
        print(f" -> Додавання користувача: {user_name} (ID: {label_id})")
        
        for image_file in os.listdir(user_path):
            # Перевірка розширення файлу
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_path = os.path.join(user_path, image_file)
            
            # Читання зображення в градаціях сірого (обов'язкова умова для LBPH)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Помилка: Не вдалося завантажити зображення {image_path}")
                continue
                
            # Переконаємось, що розмір зображення достатній 
            if img.shape[0] < 50 or img.shape[1] < 50:
                print(f"Попередження: Пропуск малого зображення {image_file}.")
                continue

            faces.append(img)
            labels.append(label_id)
            
        label_id += 1
        
    if not faces:
        print("Помилка: Не знайдено жодного зображення для тренування. Перевірте папки!")
        sys.exit()

    print(f"\nЗнайдено {len(faces)} зображень. Початок тренування...")
    
    # Тренування моделі
    recognizer.train(faces, np.array(labels))
    recognizer.write(MODEL_FILE) 
    print(f"✅ Модель LBPH навчена та збережена як {MODEL_FILE}")
    print(f"\n📢 УВАГА: Запам'ятайте цей мапінг для файлу 'recognizer_lbph.py'!")
    print(f"Мапінг імен: {names}")
    
    # Зберігання мапінгу у текстовому файлі
    with open('name_mapping.txt', 'w', encoding='utf-8') as f:
        f.write(str(names))

if __name__ == '__main__':
    train_recognizer()