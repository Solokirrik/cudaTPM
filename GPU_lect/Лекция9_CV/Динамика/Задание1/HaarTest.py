import numpy as np
import cv2
import argparse

CASCADE = "./haar-face.xml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_image', type=str, help='input video path', default='image.jpg')
    parser.add_argument('--in_cascade', type=str, help='input cascade path', default=CASCADE)

    args = parser.parse_args()

    img = cv2.imread(args.in_image)
    haar_cascade = cv2.CascadeClassifier(args.in_cascade)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detects = haar_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in detects:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = frame[y:y+h, x:x+w]

    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
    
