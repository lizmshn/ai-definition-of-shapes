import sys
import numpy as np
import cv2
import imutils
image = cv2.imread('one.png')
resized = imutils.resize(image, width=1000)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
cv2.CHAIN_APPROX_SIMPLE)
samples = np.empty((0, 100))
responses = []
keys = [i for i in range(48, 58)]
for cnt in contours:
 if cv2.contourArea(cnt) > 50:
 [x, y, w, h] = cv2.boundingRect(cnt)
 if h > 30:
 cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
 roi = thresh[y:y + h, x:x + w]
 print(roi)
 roismall = cv2.resize(roi, (10, 10))
 cv2.imshow('roi', roismall)
 cv2.imshow('image', resized)
 key = cv2.waitKey(0)
 if key == 27: # (escape to quit)
 sys.exit()
 elif key in keys:
 responses.append(int(chr(key)))
 sample = roismall.reshape((1, 100))
 samples = np.append(samples, sample, 0)
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1)) # ?
np.savetxt('generalsamples.data', samples)
np.savetxt('generalresponses.data', responses)
