import os
import sys

import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import GRU, Bidirectional, Dense, Dropout, TimeDistributed
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.utils import to_categorical

import iofn

# ========== Hyper
BATCHSIZE = 32
EPOCHS = 19
MODELDIR = 'model/keras-rnn-16/'
MODELFILE = MODELDIR + 'model'

# ========== Custom
# ========== Model
if os.path.isfile(MODELFILE):
    model = load_model(MODELFILE)
else:
    model = Sequential()

    model.add(Bidirectional(GRU(384, return_sequences=True), input_shape=(777, 39)))
    model.add(Bidirectional(GRU(384, return_sequences=True)))
    model.add(Bidirectional(GRU(384, return_sequences=True)))
    model.add(Bidirectional(GRU(384, return_sequences=True)))

    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(TimeDistributed(Dense(39, activation='softmax')))
    model.add(Dropout(0.5))

    opt = RMSprop(0.001)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()



# ========== Main
mode = sys.argv[1]
inputDir = sys.argv[2]
if mode == "train":
    data, labels = iofn.dataAndLabels(inputDir)
    labels_oh = np.reshape(to_categorical(labels, num_classes=39), [-1, 777, 39])
    tb = TensorBoard(log_dir=MODELDIR, histogram_freq=0, write_graph=True, write_grads=True)

    model.fit(data, labels_oh, epochs=EPOCHS, batch_size=BATCHSIZE, validation_split=0.1, shuffle=True, callbacks=[tb])
    model.save(MODELFILE)

if mode == "eval":
    data, labels = iofn.dataAndLabels(inputDir)
    labels_oh = np.reshape(to_categorical(labels, num_classes=39), [-1, 777, 39])

    loss = model.evaluate(data, labels_oh, batch_size=BATCHSIZE)
    print(sum(loss))

if mode == "test":
    outputFile = sys.argv[3]
    # data, labelorder = iofn.data(inputDir)
    data, labelorder, seqlen = iofn.moreData(inputDir)

    predictions = model.predict(data, batch_size=BATCHSIZE)
    predictions = np.argmax(predictions, axis=2)
    # for row in predictions:
    #     print(row)
    output = [[x[0], iofn.advTrimOutput(x[1][:x[2]])] for x in zip(labelorder, predictions, seqlen)]
    iofn.saveOutput(output, outputFile, True)
