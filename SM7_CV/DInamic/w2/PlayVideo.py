import numpy as np
import cv2
import argparse

CASCADE = "./haar-hand.xml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_video', type=str, help='input video path', default='./OpenClose2.avi')
    parser.add_argument('--in_cascade', type=str, help='input cascade path', default=CASCADE)

    args = parser.parse_args()
    cap = cv2.VideoCapture(args.in_video)
    haar_cascade = cv2.CascadeClassifier(args.in_cascade)

    if cap.isOpened():
        ret, frame = cap.read()
        frame_h, frame_w,deep = frame.shape
        frame2 = np.zeros((frame_h, frame_w, 3), np.uint8) + 0
        heat_map = np.zeros(frame2.shape)

        heat_fr = 3
        heat_eps = 1
        step_cntr = 0
        d_default = frame_w // 4
        cyan_prm = (255, 255, 0)
        white_prm = (255, 255, 255)

        hand_crd = np.zeros((1, 2), np.int32)
        hand_crd = np.delete(hand_crd, 0, 0)
        cleared = False

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                break

            detects = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6,
                                                    minSize=(d_default, d_default))
            step_cntr += 1
            for (x, y, w, h) in detects:
                cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
                heat_map[y:y + h:, x:x + w] += 1
            if step_cntr == heat_fr:
                heat_max = heat_map.max()
                # cv2.imshow('heatmap', heat_map)
                if heat_max > heat_eps:
                    hand_mean = np.argwhere(heat_map == heat_map.max()).mean(0, dtype=np.int32).reshape((1, 3))[:,:2]
                    if cleared:
                        # hand_crd = np.vstack([hand_crd, hand_mean])
                        hand_len = len(hand_crd)
                        d_step = np.sqrt(((hand_mean - hand_crd[hand_len - 1]) ** 2).sum())
                        if d_default > d_step:
                            hand_crd = np.vstack([hand_crd, hand_mean])
                            hand_len = len(hand_crd)
                            yc, xc = hand_crd[hand_len - 1]
                            y0, x0 = hand_crd[hand_len - 2]
                            cv2.line(frame2, (x0, y0), (xc, yc), cyan_prm, 2)
                            cv2.circle(frame2, (xc, yc), 2, white_prm, 2)
                            if xc > frame_w // 2:
                                cv2.rectangle(gray, (frame_w // 2, 0), (frame_w, frame_h), white_prm, 2)
                            else:
                                cv2.rectangle(gray, (0, 0), (frame_w // 2, frame_h), white_prm, 2)
                    else:
                        hand_crd = np.vstack([hand_crd, hand_mean])
                        cleared = True
                heat_map = heat_map * 0
                step_cntr = 0

            cv2.imshow('frame', gray)
            cv2.imshow('track', frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        print("Video file not found")



    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
