import numpy as np
import csv
import cv2
from os import listdir
from io import StringIO
from os.path import isfile, join


def normalize(x):
    mean = np.mean(x, dtype=np.float64).astype(x.dtype)
    std = np.std(x, dtype=np.float64).astype(x.dtype)
    return (x.astype(np.float_) - mean) / std


def load_images(txtpath, picpath):


    def form_dict(txtpath):
        mydict = {}
        with open(txtpath, 'r') as f:
            next(f) # skip header
            for row in f:
                row = row.split("\t")
                mydict[row[0]] = int(row[1])
        f.close()
        return mydict

    def form_list(f_dict, picpath):
        data_list = list()
        for item in f_dict:
            for value in range(1, f_dict[item]):
                photo_name = item + "_" + '{:04}'.format(value) + '.jpg'
                joinoo = join(picpath, item, photo_name)
                data_list.append(joinoo)
        return data_list

    def form_arr(data_list):
        images = []
        for image_file in data_list:
            img = cv2.imread(image_file, 0)
            img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_CUBIC)
            img = normalize(img)
            # rawimg = np.copy(img).astype('uint8')
            images.append(img)
        return images

    return form_arr(form_list(form_dict(txtpath), picpath))

path = './People/peopleDevTest.txt'
path_to_photo_folder = './People/lfw-deepfunneled/'
prep_people = load_images(path, path_to_photo_folder)


def predict(X_in):
    pass