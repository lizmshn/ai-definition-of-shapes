# ai-definition-of-shapes
Определение сложных фигур с реального изображения
Данный проект реализует обработку изображений с использованием OpenCV, Imutils и NumPy. В рамках работы выполняется распознавание и классификация геометрических фигур с применением алгоритма kNN.

1. Чтение и предобработка изображения
Преобразование изображения в оттенки серого.</br>
Применение размытия Гаусса.</br>
Бинаризация изображения с помощью пороговой обработки.</br>
2. Поиск контуров
3. Обработка контуров
Выделение областей интереса.
Определение размеров фигуры.
Классификация и сохранение данных для kNN.
4. Сохранение данных
