import cv2
import numpy as np
import argparse

wh_ptr = (255, 255, 255)
gr_ptr = (40,150,40)
rd_ptr = (10,140,170)

class Maze():
    def __init__(self, step=100):
        self.step = step
        self.ends = list()
        self.actv_nodes = dict()
        self.dltd_nodes = list()

    def end_points(self, gimg):
        circles = cv2.HoughCircles(gimg, cv2.HOUGH_GRADIENT, 1, 90,
                                   param1=50, param2=20, minRadius=0, maxRadius=150)
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            self.ends.append(i[:2])

    def check_movement(self, img):
        pass

    def move(self):
        pass

    def add_point(self):
        pass

    def check_short(self):
        pass

    def solve(self, img):
        img = cv2.medianBlur(img, 5)
        gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        self.end_points(gimg)

        for i in self.ends:
            cv2.circle(img, (i[0], i[1]), 20, gr_ptr, 2)
            cv2.circle(img, (i[0], i[1]), 2, rd_ptr, 3)

        cv2.circle(img, (48, 48 + self.step), 20, wh_ptr, 2)
        cv2.circle(img, (48 + self.step, 48), 20, wh_ptr, 2)
        im_shape = img.shape[:2]

        print(im_shape)

        cv2.imshow('Circles', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_image', type=str, help='input image path', default='./pic/Labyrinth_1.png')

    args = parser.parse_args()
    img = cv2.imread(args.in_image)

    path = Maze(step=100)
    path.solve(img)

    # dictoo = dict()
    # dictoo[1] = []
    # dictoo[2] = []
    # dictoo[1].append(10)
    # dictoo[1].append(1)
    # dictoo[1].append([48, 50])
    # print(dictoo)
    # print(dictoo[1][2][0])


if __name__ == "__main__":
    main()
