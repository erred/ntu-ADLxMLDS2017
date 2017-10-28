import os
import sys

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import (AveragePooling1D, Conv1D, Dense, Dropout,
                          GlobalMaxPooling1D, MaxPooling1D, TimeDistributed)
from keras.models import Sequential, load_model
from keras.optimizers import Adagrad
from keras.utils import to_categorical

import iofn

# ========== Hyper
BATCHSIZE = 64
EPOCHS = 17
MODELDIR = 'model/keras-cnn-1/'
MODELFILE = MODELDIR + 'model'

# ========== Custom
# ========== Model
if os.path.isfile(MODELFILE):
    model = load_model(MODELFILE)
else:
    model = Sequential()

    model.add(TimeDistributed(Conv1D(128, 5, strides=1, padding='same', activation='relu'), input_shape=(777, 39, 1)))
    model.add(TimeDistributed(MaxPooling1D(3, 3, 'same')))
    model.add(TimeDistributed(Conv1D(64, 7, strides=1, padding='same', activation='relu')))
    model.add(TimeDistributed(AveragePooling1D(3, 3, 'same')))
    model.add(TimeDistributed(Conv1D(32, 7, strides=1, padding='same', activation='relu')))
    model.add(TimeDistributed(GlobalMaxPooling1D()))
    model.add(TimeDistributed(Dropout(0.3)))

    model.add(TimeDistributed(Dense(128, activation='relu')))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(39, activation='softmax')))

    opt = Adagrad()
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()



# ========== Main
mode = sys.argv[1]
inputDir = sys.argv[2]
if mode == "train":
    data, labels = iofn.dataAndLabels(inputDir)
    data_ed = np.expand_dims(data, axis=3)
    labels_oh = np.reshape(to_categorical(labels, num_classes=39), [-1, 777, 39])
    tb = TensorBoard(log_dir=MODELDIR, histogram_freq=0, write_graph=True, write_grads=True)

    model.fit(data_ed, labels_oh, epochs=EPOCHS, batch_size=BATCHSIZE, validation_split=0.1, shuffle=True, callbacks=[tb])
    model.save(MODELFILE)

if mode == "eval":
    data, labels = iofn.dataAndLabels(inputDir)
    data_ed = np.expand_dims(data, axis=2)
    labels_oh = np.reshape(to_categorical(labels, num_classes=39), [-1, 777, 39])

    loss = model.evaluate(data_ed, labels_oh, batch_size=BATCHSIZE)
    print(sum(loss))

if mode == "test":
    outputFile = sys.argv[3]
    data, labelorder = iofn.data(inputDir)
    data_ed = np.expand_dims(data, axis=2)

    predictions = model.predict(data_ed, batch_size=BATCHSIZE)
    predictions = np.argmax(predictions, axis=2)
    # for row in predictions:
    #     print(row)
    output = [[x[0], iofn.trimOutput(x[1])] for x in zip(labelorder, predictions)]
    iofn.saveOutput(output, outputFile, True)
