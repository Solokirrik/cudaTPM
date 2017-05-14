import numpy as np
import cv2
import argparse

CASCADE = "./haar-hand.xml"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_video', type=str, help='input video path', default='Test2.avi')
    parser.add_argument('--in_cascade', type=str, help='input cascade path', default=CASCADE)

    args = parser.parse_args()
    cap = cv2.VideoCapture(args.in_video)
    haar_cascade = cv2.CascadeClassifier(args.in_cascade)

    while(cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detects = haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        for (x, y, w, h) in detects:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
