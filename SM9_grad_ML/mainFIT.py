import time
import numpy as np
import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import rectify, sigmoid, linear, tanh
from lasagne.layers import InputLayer, DenseLayer, BatchNormLayer, Upscale2DLayer, NonlinearityLayer, ReshapeLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, dropout

import matplotlib.pyplot as plt

import gzip
import pickle

from six.moves import cPickle
from matplotlib import cm
from imutils import face_utils
import os
import cv2
import dlib
from scipy import misc

np.random.seed(42)

params0 = {'legend.fontsize': 'medium',
          'figure.figsize': (20, 10),
          'agg.path.chunksize' : 0 ,
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'medium',
          'ytick.labelsize':'medium'}
plt.rcParams.update(params0)

root_dir = './People/lfw2'


def blur(img):
    return cv2.blur(img, (5, 5))


def sharp(img):
    return cv2.filter2D(img, -1, np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))


def normalize(x):
    mean = np.mean(x, dtype=np.float64).astype(x.dtype)
    std = np.std(x, dtype=np.float64).astype(x.dtype)
    return (x.astype(np.float_) - mean) / std


def get_samples(path):
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


X_train, y_train = get_samples('./People/pairsDevTrain.txt')
X_val, y_val = get_samples('./People/pairsDevTest.txt')

input_image_left  = T.tensor4('input_left')
input_image_positive = T.tensor4('input_positive')
input_image_negative = T.tensor4('input_negative')

targ_var = T.dvector("target")

l_input = InputLayer(shape=(None, 1, 250, 250), input_var=input_image_left)
p_input = InputLayer(shape=(None, 1, 250, 250), input_var=input_image_positive)
n_input = InputLayer(shape=(None, 1, 250, 250), input_var=input_image_negative)

my_nonlin = rectify
nn_l_conv1 = Conv2DLayer(l_input, 32, (3, 3), nonlinearity=my_nonlin, W=lasagne.init.GlorotUniform())
nn_l_pool1 = MaxPool2DLayer(nn_l_conv1, (2, 2))
nn_l_conv2 = Conv2DLayer(nn_l_pool1, 32, (3, 3), nonlinearity=my_nonlin)
nn_l_pool2 = MaxPool2DLayer(nn_l_conv2, (2, 2))
nn_l_dense = DenseLayer(dropout(nn_l_pool2, p=.5), num_units=256, nonlinearity=my_nonlin)
nn_l_out = DenseLayer(dropout(nn_l_dense, p=.5), num_units=128, nonlinearity=my_nonlin)

l_params = lasagne.layers.get_all_params(nn_l_out)

nn_p_conv1 = Conv2DLayer(p_input, 32, (3, 3), nonlinearity=my_nonlin, W=l_params[0], b=l_params[1])
nn_p_pool1 = MaxPool2DLayer(nn_p_conv1, (2, 2))
nn_p_conv2 = Conv2DLayer(nn_p_pool1, 32, (3, 3), nonlinearity=my_nonlin, W=l_params[2], b=l_params[3])
nn_p_pool2 = MaxPool2DLayer(nn_p_conv2, (2, 2))
nn_p_dense = DenseLayer(dropout(nn_p_pool2, p=0.5), num_units=256, nonlinearity=my_nonlin, W=l_params[4], b=l_params[5])
nn_p_out = DenseLayer(dropout(nn_p_dense, p=0.5), num_units=128, nonlinearity=my_nonlin, W=l_params[6], b=l_params[7])

nn_n_conv1 = Conv2DLayer(n_input, 32, (3, 3), nonlinearity=my_nonlin, W=l_params[0], b=l_params[1])
nn_n_pool1 = MaxPool2DLayer(nn_n_conv1, (2, 2))
nn_n_conv2 = Conv2DLayer(nn_n_pool1, 32, (3, 3), nonlinearity=my_nonlin, W=l_params[2], b=l_params[3])
nn_n_pool2 = MaxPool2DLayer(nn_n_conv2, (2, 2))
nn_n_dense = DenseLayer(dropout(nn_n_pool2, p=0.5), num_units=256, nonlinearity=my_nonlin, W=l_params[4], b=l_params[5])
nn_n_out = DenseLayer(dropout(nn_n_dense, p=0.5), num_units=128, nonlinearity=my_nonlin, W=l_params[6], b=l_params[7])

nn_merge = lasagne.layers.concat([nn_l_out, nn_p_out, nn_n_out], axis=1)

nn_out  = lasagne.layers.get_output(nn_merge, deterministic=True)
nn_out_test  = lasagne.layers.get_output(nn_merge, deterministic=True)
nn_out_left = nn_out[:, :128]
nn_out_positive = nn_out[:, 128:256]
nn_out_negative = nn_out[:, 256:]

nn_out_left_test = nn_out_test[:, :128]
nn_out_positive_test = nn_out_test[:, 128:256]
nn_out_negative_test = nn_out_test[:, 256:]

a = T.scalar()

d1 = T.sum(T.sqr(nn_out_left - nn_out_positive), axis=1)
d2 = T.sum(T.sqr(nn_out_left - nn_out_negative), axis=1)

loss = T.sum(T.maximum(T.sqr(d1) - T.sqr(d2) + a, 0.))

d1_test = T.sum(T.sqr(nn_out_left_test - nn_out_positive_test), axis=1)
d2_test = T.sum(T.sqr(nn_out_left_test - nn_out_negative_test), axis=1)

test_loss = T.sum(T.maximum(T.sqr(d1_test) - T.sqr(d2_test) + a, 0.))
# margin = 1.2
# d = T.sum(T.sqr(nn_out_left - nn_out_right), axis=1)
# d_test = T.sum(T.sqr(nn_out_left_test - nn_out_right_test), axis=1)

# loss = T.mean(targ_var * T.sqr(d) + (1 - targ_var) * T.sqr(T.maximum(margin - d, 0)))

# mean_accuracy = T.mean(T.eq(targ_var, (d_test < margin)))

params = lasagne.layers.get_all_params(nn_merge)
# updates = lasagne.updates.rmsprop(loss, params)
updates = lasagne.updates.adamax(loss, params)
# updates = lasagne.updates.nesterov_momentum(loss, params, 0.01)

train_fn = theano.function([input_image_left, input_image_positive, input_image_negative, a], loss,
                           updates=updates, allow_input_downcast=True)
val_fn = theano.function([input_image_left, input_image_positive, input_image_negative, a], test_loss,
                         updates=updates, allow_input_downcast=True)
test_fn = theano.function([input_image_left, input_image_positive, input_image_negative], [d1_test, d2_test],
                          allow_input_downcast=True)
output_fn = theano.function([input_image_left, input_image_positive, input_image_negative], nn_out,
                            allow_input_downcast=True)


def iterate_minibatches(inputs, targets, batchs_per_epoch=100, batchsize=20, train=True, shuffle=False):
    assert len(inputs) == len(targets)

    left_indices = np.arange(len(inputs))

    if shuffle:
        np.random.shuffle(left_indices)

    for _ in range(batchs_per_epoch):
        full_lft_indxs = []
        full_pos_indxs = []
        full_neg_indxs = []

        for _ in range(batchsize):
            start_idx = np.random.randint(low=0, high=len(left_indices))
            full_lft_indxs.append(start_idx)

            pos_idxs = np.where(targets == targets[start_idx])[0]
            b_idxs = np.random.randint(low=0, high=len(pos_idxs), size=1)
            full_pos_indxs.append(pos_idxs[b_idxs[0]])

            neg_idxs = np.where(targets != targets[start_idx])[0]
            b_idxs = np.random.randint(low=0, high=len(neg_idxs), size=1)
            full_neg_indxs.append(neg_idxs[b_idxs[0]])

        full_lft_indxs = np.array(full_lft_indxs)
        full_pos_indxs = np.array(full_pos_indxs)
        full_neg_indxs = np.array(full_neg_indxs)

        yield inputs[full_lft_indxs], inputs[full_pos_indxs], inputs[full_neg_indxs]

num_epochs = 200
train_errors = []
val_errors = []
epoch = 0
batch_size = 50
batchs_per_epoch = 20
margin = 1.242

for epoch in range(epoch, num_epochs):

    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, batchs_per_epoch=batchs_per_epoch,
                                     batchsize=batch_size, train=True, shuffle=True):
        inputs_left, inputs_positive, inputs_negative = batch
        err = train_fn(inputs_left, inputs_positive, inputs_negative, margin)
        train_err += err
        train_batches += 1
    print(err)

    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, batchs_per_epoch=batchs_per_epoch,
                                     batchsize=batch_size, train=False, shuffle=True):
        inputs_left, inputs_positive, inputs_negative = batch
        err = val_fn(inputs_left, inputs_positive, inputs_negative, margin)
        val_err += err
        val_batches += 1

    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    train_errors.append(train_err / train_batches)
    val_errors.append(val_err / val_batches)

plt.plot(np.log(train_errors), 'r')
plt.plot(np.log(val_errors), 'b')
plt.show()

filename = open('nn_param3.dat', 'wb')
cPickle.dump(output_fn, filename, protocol=cPickle.HIGHEST_PROTOCOL)
filename.close()

# filename = open('nn_param2.dat', 'rb')
# output_fn = pickle.load(filename)
# filename.close()

for i in range(X_val.shape[0]):
    res = output_fn([X_val[i]], [X_val[0]], [X_val[0]])
    x1 = res[:, :128]
    x2 = res[:, 128:256]
    x3 = res[:, 256:]
    print(y_val[i], ((x1-x3)**2).sum())

for i in range(X_train.shape[0]):
    res = output_fn([X_train[i]], [X_train[1]], [X_train[1]])
    x1 = res[:, :128]
    x2 = res[:, 128:256]
    x3 = res[:, 256:]
    print(y_train[i], ((x1-x3)**2).sum())

# res = output_fn([X_val[665]], [X_val[1]], [X_val[1]])
# print(res[:, :128])
# print(res[:, 128:256])
# print(res[:, 256:])
