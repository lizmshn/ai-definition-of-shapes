import cv2
import imutils
import numpy as np
data = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))
model = cv2.ml.KNearest_create()
model.train(data, cv2.ml.ROW_SAMPLE, responses)
color_ranges = {
 'blue': (np.array([95, 100, 100]), np.array([135, 255, 255])),
 'green': (np.array([36, 40, 40]), np.array([90, 255, 255])),
 'red': (np.array([0, 100, 100]), np.array([12, 255, 255]), np.array([164, 100, 100]),
np.array([180, 255, 255]))
}
figures = {
 1: 'star',
 2: 'heart',
 3: 'circle',
 4: 'triangle',
 5: 'square',
}
colors_bgr = {
 'blue': (255, 0, 0),
 'green': (0, 255, 0),
 'red': (0, 0, 255),
}
image = cv2.imread('two.jpg')
resized_image = imutils.resize(image, width=1000)
processed_image = resized_image.copy()
cv2.imshow('image', processed_image)
result_image = np.ones((resized_image.shape[0], resized_image.shape[1], 3),
dtype=np.uint8) * 255
def check_colors():
 blurred = cv2.GaussianBlur(resized_image, (5, 5), 0)
 hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
 for color_name, color_range in color_ranges.items():
 mask = cv2.inRange(hsv, color_range[0], color_range[1])
 if color_name == 'red':
 mask += cv2.inRange(hsv, color_range[2], color_range[3])
 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)[-2:]
 contours = [i for i in contours if cv2.contourArea(i) > 50]
 find_figure(contours, mask, color_name)
def find_figure(contours, mask, color_name):
 for cnt in contours:
 x, y, w, h = cv2.boundingRect(cnt)
 if h > 30:
 try:
 cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
 cv2.drawContours(result_image, [cnt], 0, colors_bgr[color_name], -1)
 roi = mask[y:y + h, x:x + w]
 aspect_ratio = float(w) / h
 resized_roi = cv2.resize(roi, (10, 10))
 resized_roi = resized_roi.reshape((1, 100))
 resized_roi = np.float32(resized_roi)
 _, results, _, _ = model.findNearest(resized_roi, k=1)
 figure_num = int(results[0])
 figure_type = figures[figure_num]
 text = "{} {}".format(color_name, figure_type)
 cv2.putText(result_image, text, (x + w // 2, y + h // 2),
cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 179, 118), 2)
 except cv2.Error as e:
 print('Invalid')
cv2.putText(result_image, '1142 Mishina Elizaveta', (80, 730),
cv2.FONT_HERSHEY_SIMPLEX, 1, (22, 179, 118), 2)
check_colors()
cv2.imshow('result', result_image)
cv2.waitKey(0)
