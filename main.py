import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('image.jpg')
new_image = np.zeros(image.shape, dtype='uint8')

# Преобразование в оттенки серого
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.GaussianBlur(gray_image, (1, 1), 0)

gray_image = cv2.Canny(gray_image, 30, 70)

# Находим контуры
contours, hierarchy = cv2.findContours(gray_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Определяем количество горизонтальных областей
num_sections = 3

# Получаем размеры изображения
height, _, _ = image.shape

# Вычисляем высоту каждой секции
section_height = height // num_sections

# Определяем верхние и нижние границы каждой секции
section_boundaries = [(i * section_height, (i + 1) * section_height) for i in range(num_sections)]

# Разделяем контуры на три набора соответствующих трем горизонтальным областям
contours_by_section = [[] for _ in range(num_sections)]
for contour in contours:
    # Находим центр контура
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Определяем секцию, к которой относится центр контура
    section_index = min(max(cY // section_height, 0), num_sections - 1)

    # Добавляем контур в соответствующий набор
    contours_by_section[section_index].append(contour)

#  Уникальный цвет для каждой секции
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# Рисуем контуры с разными цветами для каждой секции
for i, contours_in_section in enumerate(contours_by_section):
    color = colors[i % len(colors)]  # Выбираем уникальный цвет для текущей секции
    cv2.drawContours(new_image, contours_in_section, -1, color, 1)  # Рисуем контуры текущей секции

# Сохраняем изображение в файл
cv2.imwrite('result.jpg', new_image)

cv2.imshow('result', new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
