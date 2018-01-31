from keras.layers import Conv2D, BatchNormalization, Input, Dropout
from keras.layers import LeakyReLU, Flatten, Reshape, Concatenate, Dense

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize
from skimage.color import rgb2grey
import numpy as np
import pickle
import os
import sys

EPOCHS = 20
BATCHSIZE = 16

INPUT_SIZE = 256
CLASSES = 15

TRAINSIZE = 111240
STEPSPEREPOCH = 6954
VALIDATIONSTEPS = 28

MODELDIR = 'model/' + os.path.basename(os.path.splitext(sys.argv[0])[0])
MODELFILE = MODELDIR + '/model'

tb = TensorBoard(log_dir=MODELDIR)
sv = ModelCheckpoint(MODELFILE, save_best_only=True, save_weights_only=True)
es = EarlyStopping(patience=3)


def main():
    model = create_model()
    if os.path.exists(MODELFILE):
        model.load_weights(MODELFILE)

    model.fit_generator(
        labelgen(),
        steps_per_epoch=STEPSPEREPOCH,
        validation_data=validgen(),
        validation_steps=VALIDATIONSTEPS,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=[tb, sv, es])


def create_model():
    def conv(xi, kernel, filters, strides=1):
        x = Conv2D(filters, kernel, padding='same')(xi)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Concatenate()([x, xi])
        return x

    xi = Input([INPUT_SIZE, INPUT_SIZE])
    x = Reshape([INPUT_SIZE, INPUT_SIZE, 1])(xi)

    x = conv(x, 3, 8)
    x = conv(x, 3, 8)
    x = conv(x, 3, 8)
    x = conv(x, 3, 8)
    x = conv(x, 3, 8)
    x = conv(x, 3, 8)
    x = Conv2D(48, 1, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = conv(x, 3, 12)
    x = conv(x, 3, 12)
    x = conv(x, 3, 12)
    x = conv(x, 3, 12)
    x = conv(x, 3, 12)
    x = conv(x, 3, 12)
    x = Conv2D(120, 1, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = conv(x, 3, 16)
    x = conv(x, 3, 16)
    x = conv(x, 3, 16)
    x = conv(x, 3, 16)
    x = conv(x, 3, 16)
    x = conv(x, 3, 16)
    x = Conv2D(216, 1, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = conv(x, 3, 24)
    x = conv(x, 3, 24)
    x = conv(x, 3, 24)
    x = conv(x, 3, 24)
    x = conv(x, 3, 24)
    x = conv(x, 3, 24)
    x = Conv2D(360, 1, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(64, 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Flatten()((x))
    x = Dropout(0.5)(x)
    p = Dense(CLASSES, activation='softmax')(x)

    # ============ Model
    model = Model(xi, p)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


# def boxedgen():
#     with open('data/pickles/labels_boxed.pkl', 'rb') as f:
#         boxed = pickle.load(f)
#     while True:
#         for k, v in boxed.items():
#             yield img_as_float(rescale(imread('data/images/' + k),
#                                        0.25)), v, [0.25, 0.25, 0.25, 0.25]


def basegen(file):
    with open(file, 'rb') as f:
        traindata = pickle.load(f)
    while True:
        for k, v in traindata.items():
            lab = [0] * CLASSES
            for obs in v:
                lab[obs] = 1
            # x = img_as_float(
            #     resize(imread('data/images/' + k), (INPUT_SIZE, INPUT_SIZE)))
            # if len(x.shape) == 3:
            #     x = rgb2grey(x)
            x = img_as_float(imread('data/resized/' + k))
            yield x, lab


def labelgen():
    X, Y = [], []
    g = basegen('data/pickles/labels_train.pkl')
    while True:
        x, y = next(g)
        X.append(x)
        Y.append(y)
        if len(X) == BATCHSIZE:
            yield np.stack(X), np.array(Y)
            X, Y = [], []


def validgen():
    X, Y = [], []
    g = basegen('data/pickles/labels_valid.pkl')
    while True:
        x, y = next(g)
        X.append(x)
        Y.append(y)
        if len(X) == BATCHSIZE:
            yield np.stack(X), np.array(Y)
            X, Y = [], []


if __name__ == '__main__':
    main()
