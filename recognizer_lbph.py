import cv2
import sys

# --- НАЛАШТУВАННЯ ---

# Ім'я файлу навченої моделі
MODEL_FILE = "lbph_model.yml"

# Поріг впевненості (Confidence Threshold) для LBPH: 
# Чим МЕНШЕ це значення, тим СУВОРІШИЙ критерій. 
# Якщо confidence > CONFIDENCE_THRESHOLD, особа вважається "Невідомою".
CONFIDENCE_THRESHOLD = 85 

# Індекс камери: 0 - вбудована, 1 - зовнішня (змініть, якщо потрібно)
CAMERA_INDEX = 0

# --- РУЧНЕ ОНОВЛЕННЯ ---
# ОНОВІТЬ ЦЕ! Вставте мапінг (ID: Ім'я), який ви отримали з train_lbph.py
NAME_MAPPING = {
    0: 'Іван_Сидоренко', 
    1: 'Марія_Петренко', 
    # Додайте інші ID та імена тут, якщо вони є у вашій базі!
}

# --- ІНІЦІАЛІЗАЦІЯ ---

# Завантаження Haar Cascade для детекції обличчя
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Завантаження навченої моделі
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    print(f"✅ Модель {MODEL_FILE} успішно завантажена.")
except cv2.error:
    print(f"Помилка: Не вдалося завантажити {MODEL_FILE}. Спочатку виконайте train_lbph.py!")
    sys.exit()

# --- ОСНОВНИЙ ЦИКЛ РОЗПІЗНАВАННЯ ---
def run_recognition():
    """Запускає розпізнавання облич у реальному часі."""
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Помилка: Не вдалося відкрити вебкамеру.")
        sys.exit()

    print("\n--- СИСТЕМА РОЗПІЗНАВАННЯ ЗАПУЩЕНА ---")
    print(f"Поріг впевненості: {CONFIDENCE_THRESHOLD}")
    print("Натисніть 'q' для виходу.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Попереднє перетворення для детектора
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Детекція обличчя (Вимога 2)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            
            # 3. Розпізнавання обличчя
            # recognizer.predict повертає ID та confidence (відстань).
            label_id, confidence = recognizer.predict(roi_gray)
            
            # LBPH: Чим менше confidence (відстань) до 0, тим краще збіг.
            
            # 4. Обробка результатів та візуалізація (Вимога 3)
            
            # Обчислення ймовірності (100 - confidence, щоб 100% = ідеальний збіг)
            probability = max(0, min(100, 100 - confidence))

            if confidence < CONFIDENCE_THRESHOLD:
                # Збіг знайдено: Зелений прямокутник + Ім'я
                name = NAME_MAPPING.get(label_id, "Користувач (ID не знайдено)")
                box_color = (0, 255, 0) # Зелений
                
                # Форматування тексту
                text = f"{name}"
                prob_text = f"Імовірність: {probability:.2f}%"
            else:
                # Збіг не знайдено: Червоний прямокутник + Невідомий
                name = "Невідомий"
                box_color = (0, 0, 255) # Червоний
                text = name
                prob_text = f"Імовірність: {probability:.2f}%"


            # Візуалізація прямокутника (Вимога 2, 3)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            # Візуалізація імені та імовірності
            cv2.putText(frame, text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            cv2.putText(frame, prob_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            

        # Відображення кадру
        cv2.imshow('Face Recognition System (HW2)', frame)

        # Умова виходу (натисніть 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Звільнення ресурсів
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_recognition()