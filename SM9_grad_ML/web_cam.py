import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils

CASCADE = "./cascade/haar_frontface.xml"
shapo = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapo)
# CASCADE = "./cascade/haar-face.xml"

cyan_par = (255, 255, 0)
white_par = (255, 255, 255)
red_par = (0, 0, 255)

def face_cascade(faceCascade, frame, gray):
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.05, minNeighbors=5,
                                         minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def ochorn_points(frame, gray):
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), cyan_par, 2)

        cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan_par, 2)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 3, red_par, -1)

def work_fun(video):
    faceCascade = cv2.CascadeClassifier(CASCADE)
    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade(faceCascade, frame, gray)
        # ochorn_points(frame, gray)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    cap_video = cv2.VideoCapture(0)
    if cap_video.isOpened():
        work_fun(cap_video)
    else:
        print("No camera")

    cap_video.release()
    cv2.destroyAllWindows()

main()
