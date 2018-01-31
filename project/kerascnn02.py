from keras.layers import Conv2D, BatchNormalization, Input
from keras.layers import LeakyReLU, Lambda, Reshape, Concatenate, Dense

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras.backend as K
from keras.models import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import pickle
from skimage.io import imread
from skimage import img_as_float
from skimage.transform import resize
from skimage.color import rgb2grey
import numpy as np
import os
import sys

TRAINSIZE = 111240

EPOCHS = 20
STEPSPEREPOCH = 1024
VALIDATIONSTEPS = 2
BATCHSIZE = 32

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

    model.fit_generator(
        labelgen(),
        steps_per_epoch=STEPSPEREPOCH,
        validation_data=validgen(),
        validation_steps=VALIDATIONSTEPS,
        epochs=EPOCHS,
        workers=3,
        use_multiprocessing=True,
        shuffle=True,
        callbacks=[tb, sv, es])


def create_model():

    xi = Input([INPUT_SIZE, INPUT_SIZE])
    x = Reshape([INPUT_SIZE, INPUT_SIZE, 1])(xi)

    ir = InceptionResNetV2(
        include_top=False,
        weights=None,
        input_tensor=x,
        input_shape=(INPUT_SIZE, INPUT_SIZE, 1),
        pooling='avg')

    p = Dense(CLASSES, activation='softmax')(ir.output)

    # ============ Model
    model = Model(ir.input, p)
    model.compile('adam', 'binary_crossentropy')
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
            x = img_as_float(
                resize(imread('data/images/' + k), (INPUT_SIZE, INPUT_SIZE)))
            if len(x.shape) == 3:
                x = rgb2grey(x)
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
