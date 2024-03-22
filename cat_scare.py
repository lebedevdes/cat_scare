"""Пугаем котов поливалками"""
import cv2
from ultralytics import YOLO


# Укажите путь к вашему видеофайлу
video_path = 'video/video_02.mp4'

# Загрузка видеофайла
cap = cv2.VideoCapture(0)
model = YOLO("yolov8m.pt")

# Проверка, успешно ли открылся файл
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видеофайл.")
    exit()

# Цикл для чтения кадров из видео
while True:
    # Чтение кадра
    ret, frame = cap.read()

    # Если кадр был прочитан корректно ret=True
    if not ret:
        print("Не удалось прочитать видео или видео закончилось. Выход из цикла.")
        break

    prediction = model(frame)[0]
    frame = prediction.plot(img=frame)

    # Отображение кадра
    cv2.imshow('Видео', frame)

    # Ожидание нажатия клавиши 'q' для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие всех окон
cap.release()
cv2.destroyAllWindows()
