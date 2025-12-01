import cv2
import os
import sys

# --- НАЛАШТУВАННЯ ---

# 1. ***ЗМІНІТЬ ЦЕ ІМ'Я!*** # Використовуйте нижнє підкреслення (_) замість пробілів для уникнення конфліктів.
USER_NAME = "Марія_Петренко" 

# 2. Шлях до папки бази даних
OUTPUT_DIR = os.path.join("faces_db", USER_NAME)

# Індекс камери: 0 - зазвичай вбудована, 1 - зовнішня (змініть, якщо потрібно)
CAMERA_INDEX = 0 

# --- ФУНКЦІЯ ВИЯВЛЕННЯ ОБЛИЧЧЯ ---
# Використовуємо Haar Cascade для виявлення обличчя.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- ОСНОВНА ЛОГІКА ---
def initialize_capture():
    """Створює необхідні папки та ініціалізує камеру."""
    try:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Створено папку: {OUTPUT_DIR}")
        else:
            print(f"Папка вже існує: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Помилка створення папки: {e}")
        sys.exit()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Помилка: Не вдалося відкрити вебкамеру з індексом {CAMERA_INDEX}. Спробуйте змінити CAMERA_INDEX на 1.")
        sys.exit()
    
    return cap

def run_capture():
    """Основний цикл захоплення зображень."""
    cap = initialize_capture()
    img_counter = 0

    print("\n--- ІНСТРУКЦІЇ ---")
    print(f"Поточний користувач: {USER_NAME}")
    print(f"Зображення будуть збережені у: {OUTPUT_DIR}")
    print("Натисніть [ПРОБІЛ] для ЗБЕРЕЖЕННЯ обрізаного кадру.")
    print("Натисніть [Q] для ВИХОДУ.")
    print("------------------\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Детекція обличчя
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        face_detected = False
        
        for (x, y, w, h) in faces:
            # Обведення обличчя зеленим прямокутником
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_detected = True
            
            # Обрізаний кадр для збереження
            roi_color = frame[y:y + h, x:x + w]
        
        # Відображення статусу та інструкцій
        status_text = f"User: {USER_NAME} | Saved: {img_counter}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if not face_detected:
            cv2.putText(frame, "Face Not Detected! Move closer.", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Capture for Face Recognition DB", frame)

        k = cv2.waitKey(1)
        
        # Логіка збереження: k % 256 == 32 (Пробіл)
        if k % 256 == 32 and face_detected: 
            # Зберігаємо обрізаний кадр
            img_name = os.path.join(OUTPUT_DIR, f"{USER_NAME}_{img_counter:03d}.jpg")
            cv2.imwrite(img_name, roi_color)
            print(f"✅ Збережено: {img_name}")
            img_counter += 1
        
        # Логіка виходу: Q
        elif k % 256 == ord('q'):
            print("Вихід із програми...")
            break

    # Звільнення ресурсів
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_capture()