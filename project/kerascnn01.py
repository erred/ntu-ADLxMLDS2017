from keras.layers import Conv2D, BatchNormalization, Input, Activation
from keras.layers import LeakyReLU, Lambda, Reshape, Concatenate, Add

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras.backend as K
from keras.models import Model
import pickle
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import rescale
import numpy as np
import os
import sys

TRAINSIZE = 111240

EPOCHS = 20
STEPSPEREPOCH = 1024
VALIDATIONSTEPS = 2
BATCHSIZE = 64

INPUT_SIZE = 256

OUTPUT_GRID = 8
BOXES = 5
CLASSES = 15
FINAL_DIMS = BOXES * (5 + CLASSES)

MODELDIR = 'model/' + os.path.basename(os.path.splitext(sys.argv[0])[0])
MODELFILE = MODELDIR + '/model'

tb = TensorBoard(log_dir=MODELDIR)
sv = ModelCheckpoint(MODELFILE, save_best_only=True, save_weights_only=True)
es = EarlyStopping(patience=3)


def main():
    model = create_model()

    def g():
        X, Y, W = [], [], []
        while True:
            if len(X) < BATCHSIZE * 0.1:
                x, y, w = next(boxedgen())
            else:
                x, y, w = next(labelgen())
            X.append(x)
            Y.append(y)
            W.append(w)
            if len(X) == BATCHSIZE:
                Y = np.stack(Y)
                Y = [Y[..., 0:1], Y[..., 1:3], Y[..., 3:5], Y[..., 5:]]
                yield np.stack(X), Y, [
                    np.squeeze(w) for w in np.hsplit(np.array(W), 4)
                ]
                X, Y, W = [], [], []

    def vg():
        X, Y, W = [], [], []
        while True:
            x, y, w = next(boxedgen())
            # x, y = next(validgen())
            # w = [0, 0, 0, 1]
            X.append(x)
            Y.append(y)
            W.append(w)
            if len(X) == BATCHSIZE:
                Y = np.stack(Y)
                Y = [Y[..., 0:1], Y[..., 1:3], Y[..., 3:5], Y[..., 5:]]
                yield np.stack(X), Y, [
                    np.squeeze(w) for w in np.hsplit(np.array(W), 4)
                ]
                X, Y, W = [], [], []

    model.fit_generator(
        g(),
        steps_per_epoch=STEPSPEREPOCH,
        validation_data=vg(),
        validation_steps=VALIDATIONSTEPS,
        epochs=EPOCHS,
        workers=3,
        use_multiprocessing=True,
        shuffle=True,
        callbacks=[tb, sv, es])


def create_model():
    def conv(x, filters, strides=1):
        x = Conv2D(filters, 3, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    def space_to_depth(x):
        ss = x.shape[1] // 2
        ldim = x.shape[-1]
        x = K.reshape(x, [-1, ss, ss, ss, ss, ldim])
        x = K.permute_dimensions(x, [0, 1, 3, 2, 4, 5])
        x = K.reshape(x, [-1, ss, ss, ldim * 4])
        return x

    xi = Input([INPUT_SIZE, INPUT_SIZE])

    x = Reshape([INPUT_SIZE, INPUT_SIZE, 1])(xi)

    x = conv(x, 32)
    x = conv(x, 32, 2)

    x = conv(x, 32)
    x = conv(x, 32, 2)

    x = conv(x, 64)
    x = conv(x, 64, 2)

    x = conv(x, 64)
    x = conv(x, 64, 2)

    passthrough = Lambda(space_to_depth)(x)

    x = conv(x, 128)
    x = conv(x, 128, 2)
    x = conv(x, 128)

    x = Concatenate()([x, passthrough])

    x = conv(x, 256)
    x = conv(x, 256)

    x = Conv2D(FINAL_DIMS, 3, padding='same')(x)
    x = Reshape([OUTPUT_GRID, OUTPUT_GRID, BOXES, 5 + CLASSES])(x)

    # ============ Output
    conf = Lambda(lambda x: x[:, :, :, :, 0:1])(x)
    conf = Activation('sigmoid', name='c')(conf)

    xy = Lambda(lambda x: x[:, :, :, :, 1:3])(x)
    xy = Activation('sigmoid', name='xy')(xy)

    # wh = Lambda(lambda x: K.exp(x[:, :, :, :, 3:5]))(x)
    wh = Lambda(lambda x: x[:, :, :, :, 3:5])(x)
    wh = Activation('tanh')(wh)
    ones = Lambda(lambda x: K.ones_like(x))(wh)
    wh = Add(name='wh')([wh, ones])

    prob = Lambda(lambda x: x[:, :, :, :, 5:])(x)
    prob = Activation('sigmoid', name='p')(prob)

    # ============ Model
    model = Model(xi, [conf, xy, wh, prob])
    model.compile(
        'adam', ['binary_crossentropy', 'mse', 'mse', 'binary_crossentropy'],
        loss_weights=[1, 5, 5, 1])
    model.summary()
    return model


def boxedgen():
    with open('data/pickles/labels_boxed.pkl', 'rb') as f:
        boxed = pickle.load(f)
    while True:
        for k, v in boxed.items():
            yield img_as_float(rescale(imread('data/images/' + k),
                                       0.25)), v, [0.25, 0.25, 0.25, 0.25]


def labelgen():
    with open('data/pickles/labels_train.pkl', 'rb') as f:
        traindata = pickle.load(f)
    while True:
        for k, v in traindata.items():
            lab = np.zeros([OUTPUT_GRID, OUTPUT_GRID, BOXES, 5 + CLASSES])
            for obs in v:
                lab[:, :, :, obs] = 1
            yield img_as_float(rescale(imread('data/images/' + k),
                                       0.25)), lab, [0, 0, 0, 1]


def validgen():
    with open('data/pickles/labels_valid.pkl', 'rb') as f:
        traindata = pickle.load(f)
    while True:
        for k, v in traindata.items():
            lab = np.zeros([OUTPUT_GRID, OUTPUT_GRID, BOXES, 5 + CLASSES])
            for obs in v:
                lab[:, :, :, obs] = 1
            yield img_as_float(rescale(imread('data/images/' + k),
                                       0.25)), lab, [0, 0, 0, 1]


if __name__ == '__main__':
    main()
