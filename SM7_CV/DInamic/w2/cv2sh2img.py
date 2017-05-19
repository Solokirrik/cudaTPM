from time import time
from numpy import uint8
from numpy.random import rand
import cv2

xy=(512,512)
Nf = 500

def fpsopencv(dat, dat2):
    tic = time()

    for i in dat:
        cv2.imshow('test2', 255)
        cv2.imshow('test1',i)
        cv2.waitKey(10) #integer milliseconds, 0 makes wait forever
    cv2.destroyAllWindows()
    return Nf / (time()-tic)


img1 = (rand(Nf,xy[0],xy[1])*255).astype(uint8)
img2
fps = fpsopencv(img1, img2)

print(f'{fps} fps')