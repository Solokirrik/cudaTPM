import cv2
import numpy as np
from matplotlib import pyplot as plt

def normalize(x):
    mean = np.mean(x, dtype=np.float64).astype(x.dtype)
    std = np.std(x, dtype=np.float64).astype(x.dtype)
    return (x.astype(np.float_) - mean) / std

img = cv2.imread('./People/Kirill_Tishenkov/2.JPG', 1)

blur = normalize(cv2.blur(img,(5,5)))

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(img, -1, kernel)
shgray = cv2.cvtColor(sharpen, cv2.COLOR_BGR2GRAY)
sharp = normalize(shgray)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
# plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(sharp),plt.title('sharped')
plt.xticks([]), plt.yticks([])
plt.show()