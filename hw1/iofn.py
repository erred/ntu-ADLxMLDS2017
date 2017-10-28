import csv
import itertools
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import normalize

BATCHSIZE = 16
DATASET = "mfcc"
FEATURES = 39
FFTSIZE = 32
MAXLEN = 777


def saveOutput(arr, outfile, header):
    """writes arr, with header"""
    with open(outfile, 'w') as of:
        w = csv.writer(of)
        if header:
            w.writerow(["id", "phone_sequence"])
        w.writerows(arr)


phonemap = {'aa': 0, 'ae': 1, 'ah': 2, 'ao': 0, 'aw': 3, 'ax': 2, 'ay': 4, 'b': 5, 'ch': 6, 'cl': 30, 'd': 7, 'dh': 8, 'dx': 9, 'eh': 10, 'el': 20, 'en': 22, 'epi': 30, 'er': 11, 'ey': 12, 'f': 13, 'g': 14, 'hh': 15, 'ih': 16, 'ix': 16,
            'iy': 17, 'jh': 18, 'k': 19, 'l': 20, 'm': 21, 'ng': 23, 'n': 22, 'ow': 24, 'oy': 25, 'p': 26, 'r': 27, 'sh': 29, 'sil': 30, 's': 28, 'th': 32, 't': 31, 'uh': 33, 'uw': 34, 'vcl': 30, 'v': 35, 'w': 36, 'y': 37, 'zh': 29, 'z': 38}


def encodePhone(phone):
    return phonemap[phone]


alpha = "abceghiklmnrstuvwyzABCDEFGHIJKLMNOPQSTU"


def decodePhone(id):
    return alpha[id]

def noTrimOutput(arr):
    s = ''.join([decodePhone(c) for c in arr])
    s = s.strip('L')
    return s

def advTrimOutput(arr):
    s = noTrimOutput(arr)
    s = [''.join(g) for _, g in itertools.groupby(s)]
    s = ''.join([c for c in s if len(c) > 2])
    return ''.join(i for i, _ in itertools.groupby(s))


def trimOutput(arr):
    s = ''.join([decodePhone(c) for c in arr])
    s = s.strip('L')
    return ''.join(i for i, _ in itertools.groupby(s))


def readData(inputFile, mode):
    d1 = 3696
    if mode == "test":
        d1 = 592
    arr = np.zeros((d1, 777, 39))
    labels = []
    seqlen = []

    ark = np.loadtxt(
        inputFile,
        delimiter=' ',
        dtype='U20, 39f4')

    sentence = -1
    previd = 1000
    for r in ark:
        keys = r[0].split('_')
        id = int(keys[2])
        if id < previd:
            if previd < 999:
                normalize(arr[sentence, :previd], copy=False)
                seqlen.append(previd)
            sentence += 1
            labels.append(keys[0] + "_" + keys[1])
        previd = id
        arr[sentence, id - 1] = r[1]
    # last set
    normalize(arr[sentence, :previd], copy=False)
    seqlen.append(previd)

    return arr, np.array(labels, dtype='U20'), np.array(seqlen, dtype=int)


def readLabel(inputFile, labelorder):
    arr = np.full((labelorder.shape[0], 777), fill_value=30, dtype=int)
    lab = np.loadtxt(
        inputFile,
        delimiter=',',
        dtype='U20, U3')

    sentence = 0
    previd = 1000
    for r in lab:
        keys = r[0].split('_')
        id = int(keys[2])
        if id < previd:
            key = keys[0] + "_" + keys[1]
            sentence = np.where(labelorder == key)[0][0]
        previd = id
        arr[sentence, id - 1] = encodePhone(r[1])
    return arr

def data(inputDir):
    data, labelorder, seqlen = readData(
        os.path.join(inputDir, DATASET, 'test.ark'), "test")
    return data, labelorder

def moreData(inputDir):
    data, labelorder, seqlen = readData(
        os.path.join(inputDir, DATASET, 'test.ark'), "test")
    return data, labelorder, seqlen

def dataAndLabels(inputDir):
    data, labelorder, seqlen = readData(
        os.path.join(inputDir, DATASET, 'train.ark'), "train")
    label = readLabel(os.path.join(inputDir, 'label/train.lab'), labelorder)
    return data, label
