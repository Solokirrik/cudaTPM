import os
import cv2
import dlib
import time
import pickle
import numpy as np
from imutils import face_utils
from scipy import misc

import theano
import lasagne
import theano.tensor as T
from lasagne.nonlinearities import rectify, sigmoid, linear, tanh
from lasagne.layers import InputLayer, DenseLayer, BatchNormLayer, Upscale2DLayer, NonlinearityLayer, ReshapeLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, dropout

CASCADE = './cascade/haar_frontface.xml'
shapo = './cascade/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapo)

cyan_par = (255, 255, 0)
white_par = (255, 255, 255)
red_par = (0, 0, 255)
green_par = (0, 255, 0)

dumped = True
frame_size = 250
fa = face_utils.FaceAligner(predictor, desiredFaceHeight=frame_size, desiredFaceWidth=frame_size)

# video_h = './People/Sweet_Dreams480.mp4'
video_h = './People/release.mp4'

# filename = open('./proc_pic/nn_param.dat', 'rb')
# output_fn = pickle.load(filename)
# filename.close()

def blur(img):
    return cv2.blur(img, (5, 5))

def sharp(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def normalize(x):
    mean = np.mean(x, dtype=np.float64).astype(x.dtype)
    std = np.std(x, dtype=np.float64).astype(x.dtype)
    return (x.astype(np.float_) - mean) / std

def get_samples(path, root_dir):
    idx_to_name = dict()
    name_to_idx = dict()
    X = list()
    y = list()
    idx = 0
    for l in open(path).readlines()[1:]:
        fields = l.strip().split('\t')
        if len(fields) == 3:
            name, first, second = fields
            if name in name_to_idx:
                cur_idx = name_to_idx[name]
            else:
                name_to_idx[name] = idx
                cur_idx = idx
                idx += 1

            first_photo = os.path.join(root_dir, name, '{}_{:04d}.jpg'.format(name, int(first)))
            second_photo = os.path.join(root_dir, name, '{}_{:04d}.jpg'.format(name, int(second)))
            first_photo_vector = normalize(misc.imread(first_photo))
            second_photo_vector = normalize(misc.imread(second_photo))

            blured_first_photo_vector = blur(first_photo_vector)
            blured_second_photo_vector = blur(second_photo_vector)

            sharped_first_photo_vector = sharp(first_photo_vector)
            sharped_second_photo_vector = sharp(second_photo_vector)

            X.append(first_photo_vector)
            y.append(cur_idx)
            X.append(second_photo_vector)
            y.append(cur_idx)

            X.append(blured_first_photo_vector)
            y.append(cur_idx)
            X.append(blured_second_photo_vector)
            y.append(cur_idx)

            X.append(sharped_first_photo_vector)
            y.append(cur_idx)
            X.append(sharped_second_photo_vector)
            y.append(cur_idx)


        elif (fields) == 4:
            first_name, first, second_name, second = fields
            first_photo = os.path.join(root_dir, first_name, '{}_{:04d}.jpg'.format(first_name, first))
            second_photo = os.path.join(root_dir, second_name, '{}_{:04d}.jpg'.format(second_name, second))
            first_photo_vector = normalize(misc.imread(first_photo))
            second_photo_vector = normalize(misc.imread(second_photo))

            blured_first_photo_vector = blur(first_photo_vector)
            blured_second_photo_vector = blur(second_photo_vector)

            sharped_first_photo_vector = sharp(first_photo_vector)
            sharped_second_photo_vector = sharp(second_photo_vector)

            if first_name in name_to_idx:
                cur_idx = name_to_idx[first_name]
            else:
                name_to_idx[first_name] = idx
                cur_idx = idx
                idx += 1
            X.append(first_photo_vector)
            y.append(cur_idx)

            X.append(blured_first_photo_vector)
            y.append(cur_idx)

            X.append(sharped_first_photo_vector)
            y.append(cur_idx)

            if second_name in name_to_idx:
                cur_idx = name_to_idx[second_name]
            else:
                name_to_idx[second_name] = idx
                cur_idx = idx
                idx += 1

            X.append(second_photo_vector)
            y.append(cur_idx)

            X.append(blured_second_photo_vector)
            y.append(cur_idx)

            X.append(sharped_second_photo_vector)
            y.append(cur_idx)

    return np.array(X).reshape([-1, 1, 250, 250]), np.array(y)

def get_nn_out_fn():
    print('nn form started')
    input_image_left = T.tensor4('input_left')
    input_image_positive = T.tensor4('input_positive')
    input_image_negative = T.tensor4('input_negative')

    # targ_var = T.dvector("target")
    print('input image')
    l_input = InputLayer(shape=(None, 1, 250, 250), input_var=input_image_left)
    p_input = InputLayer(shape=(None, 1, 250, 250), input_var=input_image_positive)
    n_input = InputLayer(shape=(None, 1, 250, 250), input_var=input_image_negative)
    print('Input done')
    my_nonlin = rectify
    nn_l_conv1 = Conv2DLayer(l_input, 32, (3, 3), nonlinearity=my_nonlin, W=lasagne.init.GlorotUniform())
    nn_l_pool1 = MaxPool2DLayer(nn_l_conv1, (2, 2))
    nn_l_conv2 = Conv2DLayer(nn_l_pool1, 32, (3, 3), nonlinearity=my_nonlin)
    nn_l_pool2 = MaxPool2DLayer(nn_l_conv2, (2, 2))
    nn_l_dense = DenseLayer(dropout(nn_l_pool2, p=.5), num_units=256, nonlinearity=my_nonlin)
    nn_l_out = DenseLayer(dropout(nn_l_dense, p=.5), num_units=128, nonlinearity=my_nonlin)

    print('left done')
    l_params = lasagne.layers.get_all_params(nn_l_out)

    nn_p_conv1 = Conv2DLayer(p_input, 32, (3, 3), nonlinearity=my_nonlin, W=l_params[0], b=l_params[1])
    nn_p_pool1 = MaxPool2DLayer(nn_p_conv1, (2, 2))
    nn_p_conv2 = Conv2DLayer(nn_p_pool1, 32, (3, 3), nonlinearity=my_nonlin, W=l_params[2], b=l_params[3])
    nn_p_pool2 = MaxPool2DLayer(nn_p_conv2, (2, 2))
    nn_p_dense = DenseLayer(dropout(nn_p_pool2, p=0.5), num_units=256, nonlinearity=my_nonlin, W=l_params[4],
                            b=l_params[5])
    nn_p_out = DenseLayer(dropout(nn_p_dense, p=0.5), num_units=128, nonlinearity=my_nonlin, W=l_params[6],
                          b=l_params[7])
    print('positive done')
    nn_n_conv1 = Conv2DLayer(n_input, 32, (3, 3), nonlinearity=my_nonlin, W=l_params[0], b=l_params[1])
    nn_n_pool1 = MaxPool2DLayer(nn_n_conv1, (2, 2))
    nn_n_conv2 = Conv2DLayer(nn_n_pool1, 32, (3, 3), nonlinearity=my_nonlin, W=l_params[2], b=l_params[3])
    nn_n_pool2 = MaxPool2DLayer(nn_n_conv2, (2, 2))
    nn_n_dense = DenseLayer(dropout(nn_n_pool2, p=0.5), num_units=256, nonlinearity=my_nonlin, W=l_params[4],
                            b=l_params[5])
    nn_n_out = DenseLayer(dropout(nn_n_dense, p=0.5), num_units=128, nonlinearity=my_nonlin, W=l_params[6],
                          b=l_params[7])
    print('negative done')
    nn_merge = lasagne.layers.concat([nn_l_out, nn_p_out, nn_n_out], axis=1)

    with np.load('model.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(nn_merge, param_values)

    nn_out = lasagne.layers.get_output(nn_merge, deterministic=True)
    # nn_out_test = lasagne.layers.get_output(nn_merge, deterministic=True)
    # nn_out_left = nn_out[:, :128]
    # nn_out_positive = nn_out[:, 128:256]
    # nn_out_negative = nn_out[:, 256:]

    # nn_out_left_test = nn_out_test[:, :128]
    # nn_out_positive_test = nn_out_test[:, 128:256]
    # nn_out_negative_test = nn_out_test[:, 256:]
    #
    # a = T.scalar()
    # d1 = T.sum(T.sqr(nn_out_left - nn_out_positive), axis=1)
    # d2 = T.sum(T.sqr(nn_out_left - nn_out_negative), axis=1)
    # loss = T.sum(T.maximum(T.sqr(d1) - T.sqr(d2) + a, 0.))

    # d1_test = T.sum(T.sqr(nn_out_left_test - nn_out_positive_test), axis=1)
    # d2_test = T.sum(T.sqr(nn_out_left_test - nn_out_negative_test), axis=1)
    # test_loss = T.sum(T.maximum(T.sqr(d1_test) - T.sqr(d2_test) + a, 0.))

    # params = lasagne.layers.get_all_params(nn_merge)
    # updates = lasagne.updates.adamax(loss, params)

    # train_fn = theano.function([input_image_left, input_image_positive, input_image_negative, a], loss,
    #                            updates=updates, allow_input_downcast=True)
    # val_fn = theano.function([input_image_left, input_image_positive, input_image_negative, a], test_loss,
    #                          updates=updates, allow_input_downcast=True)
    # test_fn = theano.function([input_image_left, input_image_positive, input_image_negative], [d1_test, d2_test],
    #                           allow_input_downcast=True)
    output_fn = theano.function([input_image_left, input_image_positive, input_image_negative], nn_out,
                                allow_input_downcast=True)
    return output_fn

def download_db(path):
    idx_to_name = dict()
    X = list()
    y = list()
    for i, name in enumerate(os.listdir(path)):
        idx_to_name[i] = name
        human_photo_dir = os.path.join(path, name)
        for photo in os.listdir(human_photo_dir):
            frame = misc.imread(os.path.join(human_photo_dir, photo))
            gray_face = detect_to_gray(frame)
            if gray_face is not None:
                X.append(list(gray_face))
                y.append(i)
    return np.array(X), np.array(y), idx_to_name

def ochorn_points(frame, gray):
    rects = detector(gray, 1)
    for rect in rects:
        # shape = predictor(gray, rect)
        # shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        xc = x + w // 2
        yc = y + h // 2
        faceAligned = fa.align(frame, gray, rect)
        if faceAligned is not None:
            return x, y, faceAligned
        else:
            return None

def detect_to_gray(frame, coord=False):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_par = ochorn_points(frame, gray_frame)
    if detected_par is not None:
        xc, yc, faces_d = detected_par
        gray_face = normalize(cv2.cvtColor(faces_d, cv2.COLOR_BGR2GRAY))
        if coord:
            return xc, yc, gray_face
        else:
            return gray_face
    else:
        return None

def predict(face):
    # face = face.flatten()
    label_min_dist = 0
    min_dist = float("inf")
    for i in range(X.shape[0]):
        res = output_nn([[face]], [[X[i]]], [[X[i]]])
        x1 = res[:, :128]
        x2 = res[:, 128:256]
        x3 = res[:, 256:]
        new_min = ((x1 - x3) ** 2).sum()
        if new_min < min_dist:
            min_dist = new_min
            label_min_dist = y[i]
    return label_min_dist

def work_fun(video):
    ret, frame = video.read()
    start_time = time.time()
    while True:
        ret, frame = video.read()
        if ret == True:
            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            detect_param = detect_to_gray(frame, coord=True)
            if detect_param is not None:
                xc, yc, gray_face = detect_param
                cv2.imshow("resized Face #{}".format(0 + 1), gray_face)
                # if time.time() - start_time > 0.1:
                #     number = predict(gray_face)
                #     cv2.putText(frame, "{}".format(idx_to_name[number]), (xc - 10, yc - 10),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, cyan_par, 2)
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

# output_nn = get_nn()
# with np.load('model.npz') as f:
#     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
# lasagne.layers.set_all_param_values(output_nn, param_values)

output_nn = get_nn_out_fn()

if not dumped:
    X, y, idx_to_name = download_db('./People/db')
    output_x = open('./db_dump/db_X', 'wb')
    pickle.dump(X, output_x)
    output_x.close()
    output_y = open('./db_dump/db_y', 'wb')
    pickle.dump(y, output_y)
    output_y.close()
    output_idx = open('./db_dump/db_ind', 'wb')
    pickle.dump(idx_to_name, output_idx)
    output_idx.close()
else:
    db_X = open('./db_dump/db_X', 'rb')
    X = pickle.load(db_X)
    db_X.close()
    db_y = open('./db_dump/db_y', 'rb')
    y = pickle.load(db_y)
    db_y.close()
    db_ind = open('./db_dump/db_ind', 'rb')
    idx_to_name = pickle.load(db_ind)
    db_ind.close()
print('People db read')
# X_train, y_train = get_samples('./People/pairsDevTrain.txt', './People/lfw2')
print('starting')

main()
