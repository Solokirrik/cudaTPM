import numpy as np
import cv2

CASCADE = "./cascade/haar_frontface.xml"
# CASCADE = "./cascade/haar-face.xml"


def main():

    cap_video = cv2.VideoCapture(0)
    faceCascade = cv2.CascadeClassifier(CASCADE)

    while True:
        # Capture frame-by-frame
        ret, frame = cap_video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap_video.release()
    cv2.destroyAllWindows()

main()
