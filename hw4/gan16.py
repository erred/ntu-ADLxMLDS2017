import csv
import os
import random
import sys
import time
from collections import deque

import numpy as np
from keras.backend import int_shape
from keras.layers import (
    Activation, AvgPool2D, BatchNormalization, Concatenate, Conv2D,
    Conv2DTranspose, Deconv2D, Dense, Dropout, Embedding, Flatten,
    GaussianNoise, GlobalAvgPool2D, Input, LeakyReLU, RepeatVector, Reshape)
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize

# np.random.seed(int(sys.argv[2]))
np.random.seed(4096)

# https://github.com/pavitrakumar78/Anime-Face-GAN-Keras

# ============== Training Tunable
# EPOCHS          = 20
BATCHSIZE = 128
UPDATE_RATIO = 2 * 1
LOGFREQ = 10
LOGIMGFREQ = 100
LOGMODELFREQ = 200
REPLAYSAMPLEPROB = 0.1

# ============== Architecture Tunable
EMBED_HAIR = 8
EMBED_EYES = 8
NOISE_INPUT = 256

# ============== Constants
BASENAME = os.path.basename(os.path.splitext(sys.argv[0])[0])
MODELDIR = 'model/' + BASENAME
MODELFILE_GEN = MODELDIR + '/genmodel'
OUTPUTDIR = 'out/' + BASENAME
HAIR_COLORS = 12 + 1
EYE_COLORS = 11 + 1
# LABELS_GOOD = 11418
# LABELS_HAIR = 2686
# LABELS_EYES = 1607
# LEBELS_BAD  = 17720

if not os.path.exists(OUTPUTDIR):
    os.makedirs(OUTPUTDIR)
if not os.path.exists(MODELDIR):
    os.makedirs(MODELDIR)

valid_eyes = [
    'aqua eyes', 'black eyes', 'blue eyes', 'brown eyes', 'gray eyes',
    'green eyes', 'orange eyes', 'pink eyes', 'purple eyes', 'red eyes',
    'yellow eyes'
]
eInvF = {
    'aqua eyes': 13.137209302325582,
    'black eyes': 50.73952095808383,
    'blue eyes': 4.210434782608695,
    'brown eyes': 8.001416430594901,
    'gray eyes': 93.11538461538461,
    'green eyes': 8.524647887323944,
    'orange eyes': 46.81491712707182,
    'pink eyes': 24.52532561505065,
    'purple eyes': 9.745255894192065,
    'red eyes': 5.602314049586777,
    'yellow eyes': 14.205364626990779
}
valid_hair = [
    'aqua hair', 'black hair', 'blonde hair', 'blue hair', 'brown hair',
    'gray hair', 'green hair', 'orange hair', 'pink hair', 'purple hair',
    'red hair', 'white hair'
]
hInvF = {
    'aqua hair': 23.04218928164196,
    'black hair': 7.459579180509413,
    'blonde hair': 5.410441767068273,
    'blue hair': 10.7776,
    'brown hair': 5.599334995843724,
    'gray hair': 21.822894168466522,
    'green hair': 19.88976377952756,
    'orange hair': 49.28780487804878,
    'pink hair': 11.3464345873105,
    'purple hair': 15.013372956909361,
    'red hair': 19.19088319088319,
    'white hair': 23.200918484500573
}

# ==================================== Data Gen


def labelReader(path):
    good_labels, only_hair, only_eyes, bad_labels = {}, {}, {}, {}
    with open(path) as f:
        re = csv.reader(f)
        for row in re:
            tags = [t.split(':')[0] for t in row[1].split('\t')[:-1]]
            hair, eyes = [], []
            for t in tags:
                if t in valid_eyes:
                    eyes.append(valid_eyes.index(t) + 1)
                elif t in valid_hair:
                    hair.append(valid_hair.index(t) + 1)
            if len(hair) == 1 and len(eyes) == 1:
                good_labels[row[0]] = {'hair': hair, 'eyes': eyes}
            elif len(hair) == 1:
                only_hair[row[0]] = {'hair': hair, 'eyes': eyes}
            elif len(eyes) == 1:
                only_eyes[row[0]] = {'hair': hair, 'eyes': eyes}
            else:
                bad_labels[row[0]] = {'hair': hair, 'eyes': eyes}
    return good_labels, only_hair, only_eyes, bad_labels


def dataReader(path):
    datag, datah, datae, datab = [], [], [], []
    lg, lh, le, lb = labelReader(path)
    for k, v in lg.items():
        hair = v['hair'][0]
        eyes = v['eyes'][0]
        sample_weight = [1, 1, 1]
        datag.append([k, sample_weight, hair, eyes])
    for k, v in lh.items():
        hair = v['hair'][0]
        sample_weight = [1, 1, 0]
        datah.append([k, sample_weight, hair, 1])
    for k, v in le.items():
        eyes = v['eyes'][0]
        sample_weight = [1, 0, 1]
        datae.append([k, sample_weight, 1, eyes])
    for k, v in lb.items():
        sample_weight = [1, 0, 0]
        datab.append([k, sample_weight, 1, 1])
    return datag + datah + datae + datab
    # return datag


def dataGen(path):
    dat = dataReader(path)
    while True:
        np.random.shuffle(dat)
        data = []
        for k in range(len(dat) // BATCHSIZE):
            i, s, h, e = [], [], [], []
            for d in dat[k * BATCHSIZE:(k + 1) * BATCHSIZE]:
                img = resize(
                    img_as_float(imread('data/faces/' + d[0] + '.jpg')),
                    (64, 64))
                i.append(img)
                s.append(d[1])
                h.append(d[2])
                e.append(d[3])
            s = [np.squeeze(x) for x in np.hsplit(np.array(s), 3)]
            data.append([np.array(i), s, np.array(h), np.array(e)])
        for d in data:
            yield d


def fakeDataGen():
    while True:
        noise = np.random.normal(size=[BATCHSIZE, NOISE_INPUT])
        hair = np.concatenate([
            np.arange(1, HAIR_COLORS),
            np.random.randint(1, HAIR_COLORS, [BATCHSIZE - (HAIR_COLORS - 1)])
        ])
        eyes = np.random.randint(1, EYE_COLORS, [BATCHSIZE])
        yield noise, hair, eyes


# ==================================== Discriminator


def convBlock(x, filters, kernel, strides=2, bn=True):
    x = Conv2D(
        filters,
        kernel,
        strides=strides,
        padding='same',
        kernel_initializer='glorot_uniform')(x)
    if bn:
        x = BatchNormalization(momentum=0.5)(x)
    x = LeakyReLU(0.2)(x)
    return x


def build_discriminator():
    mi = Input([64, 64, 3])
    m = mi
    m = convBlock(m, 64, 4, bn=False)
    m = convBlock(m, 128, 4)
    m = convBlock(m, 256, 4)
    m = convBlock(m, 512, 4)
    m = Flatten()(m)

    valid = Dense(1, activation='sigmoid')(m)
    hair = Dense(HAIR_COLORS, activation='softmax')(m)
    eyes = Dense(EYE_COLORS, activation='softmax')(m)
    model = Model(mi, [valid, hair, eyes])
    return model


# ==================================== Generator


def deconv(x, filters, kernel, strides=2, act='lrelu', pad='same', bn=True):
    x = Conv2DTranspose(
        filters,
        kernel,
        strides=strides,
        padding=pad,
        kernel_initializer='glorot_uniform')(x)
    if bn:
        x = BatchNormalization(momentum=0.5)(x)
    if act == 'lrelu':
        x = LeakyReLU(0.2)(x)
    elif act == 'tanh':
        x = Activation('tanh')(x)
    return x


def build_generator():
    # hair
    hi = Input([1])
    he = Embedding(HAIR_COLORS, EMBED_HAIR)(hi)
    he = Flatten()(he)
    he = GaussianNoise(0.1)(he)

    # eyes
    ei = Input([1])
    ee = Embedding(EYE_COLORS, EMBED_EYES)(ei)
    ee = Flatten()(ee)
    ee = GaussianNoise(0.1)(ee)

    # noise
    ni = Input([NOISE_INPUT])
    n = Concatenate()([ni, he, ee])
    n = Reshape([1, 1, -1])(n)

    n = deconv(n, 1024, 4, strides=1, pad='valid')
    n = deconv(n, 512, 4)
    n = deconv(n, 256, 4)
    n = deconv(n, 128, 4)
    n = deconv(n, 64, 3, strides=1)
    n = deconv(n, 3, 4, bn=False)

    model = Model([ni, hi, ei], n)
    return model


def build_models():
    opt = Adam(0.0002, 0.5)

    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=opt)

    discriminator = build_discriminator()
    discriminator.compile(
        loss=[
            'binary_crossentropy', 'sparse_categorical_crossentropy',
            'sparse_categorical_crossentropy'
        ],
        loss_weights=[0.34, 0.33, 0.33],
        optimizer=opt,
        metrics=['accuracy'])

    discriminator.trainable = False

    g_in = [Input([NOISE_INPUT]), Input([1]), Input([1])]
    d_out = discriminator(generator(g_in))
    combined = Model(g_in, d_out)
    combined.compile(
        loss=[
            'binary_crossentropy', 'sparse_categorical_crossentropy',
            'sparse_categorical_crossentropy'
        ],
        loss_weights=[0.5, 0.25, 0.25],
        optimizer=opt)

    return generator, discriminator, combined


def train():
    print('building models')
    gen, disc, comb = build_models()

    print('building data gens')
    datgen = dataGen('data/tags_clean.csv')
    fgen = fakeDataGen()
    replay = deque(maxlen=BATCHSIZE * 500)

    lasttime = time.time()
    dLoss, gLoss = [], []

    print('starting training')
    for i, (imgs, sample_weights, hair, eyes) in enumerate(datgen):
        # generate fake data
        noise, fh, fe = next(fgen)
        imgs_fake = gen.predict_on_batch([noise, fh, fe])

        # labels
        valid = np.ones([BATCHSIZE]) - np.random.random_sample(BATCHSIZE) * 0.2
        fake = np.random.random_sample(BATCHSIZE) * 0.2

        # train disc
        disc.trainable = True
        dl1 = disc.train_on_batch(
            imgs, [valid, hair - 1, eyes - 1], sample_weight=sample_weights)
        dl2 = disc.train_on_batch(imgs_fake, [fake, fh - 1, fe - 1])
        dLoss.append(dl1)
        dLoss.append(dl2)
        disc.trainable = False

        for k in range(UPDATE_RATIO):
            noise, fh, fe = next(fgen)
            gl = comb.train_on_batch([noise, fh, fe], [valid, fh - 1, fe - 1])
            gLoss.append(gl)

        # logging
        if i % LOGFREQ == 0:
            print('step: {}\tdLoss: {:.2f}\tgLoss: {:.2f}'.format(
                i, np.mean(dLoss), np.mean(gLoss)))
            dLoss, gLoss = [], []

        if i % LOGIMGFREQ == 0:
            fname = OUTPUTDIR + '/out_{:03d}.jpg'.format(i // 1000)
            new_img = np.clip(np.hstack(imgs_fake[:16]), -1, 1)
            if not os.path.exists(fname):
                imsave(fname, new_img)
            else:
                prev_img = img_as_float(imread(fname))
                imsave(fname, np.vstack([prev_img, new_img]))

        if i % LOGMODELFREQ == 0:
            gen.save_weights(MODELFILE_GEN)
            curtime = time.time()
            print('saved model, time: ', curtime - lasttime)
            lasttime = curtime


def early(images):
    generator = build_generator()
    generator.compile(loss=['binary_crossentropy'], optimizer='adam')
    generator.load_weights(MODELFILE_GEN)

    n = np.random.normal(size=[images, NOISE_INPUT])
    # an = np.random.normal(size=[images, EMBED_HAIR + EMBED_EYES], scale=0.1)
    h = np.array([valid_hair.index('red hair')] * images)
    e = np.array([valid_eyes.index('green eyes')] * images)
    imgs = generator.predict_on_batch([n, h, e])
    for i, img in enumerate(imgs):
        fname = 'early/{}.jpg'.format(i)
        imsave(fname, np.clip(img, -1, 1))
    fname = 'early/all.jpg'
    allimg = np.clip(np.hstack(imgs[:]), -1, 1)
    imsave(fname, allimg)


def test():
    generator = build_generator()
    generator.compile(loss=['binary_crossentropy'], optimizer='adam')
    generator.load_weights(MODELFILE_GEN)

    with open(sys.argv[2]) as f:
        if not os.path.exists('samples'):
            os.makedirs('samples')

        re = csv.reader(f)
        for r in re:
            id = int(r[0])
            eyes = 0
            hair = 0
            for i, e in enumerate(valid_eyes):
                if e in r[1]:
                    eyes = i + 1
                    break
            for i, h in enumerate(valid_hair):
                if h in r[1]:
                    hair = i + 1
                    break

            noise = np.random.normal(size=[5, NOISE_INPUT])
            hair = np.ones(5) * hair
            eyes = np.ones(5) * hair

            imgs = generator.predict_on_batch([noise, hair, eyes])
            for j in range(len(imgs)):
                fname = 'samples/sample_{}_{}.jpg'.format(id, j + 1)
                imsave(fname, np.clip(imgs[j], -1, 1))
            # fname = 'samples/all_{}.jpg'.format(id)
            # allimg = np.clip(np.hstack(imgs[:]), -1, 1)
            # imsave(fname, allimg)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'early':
        early(5)
    elif sys.argv[1] == 'test':
        test()
