import csv
import json
import os
import pickle

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import text_to_word_sequence

import preprocess

MAX_CAPTION_LEN = 45

with open('data/w2e.pkl', 'rb') as fo:
    w2e = pickle.load(fo)
def encodeCaption(sentence):
    """encodes caption words -> [int]"""
    low = text_to_word_sequence(sentence)
    return [w2e[w] for w in low]

with open('data/e2w.pkl', 'rb') as fo:
    e2w = pickle.load(fo)
def decodeCaption(loe):
    """decodes captions [int] -> words"""
    return ' '.join([e2w[w] for w in loe])

def saveOutput(arr, fname):
    with open(fname, 'w') as fo:
        writer = csv.writer(fo)
        writer.writerows(arr)

def trainEvalFromTFRecord(inputDir, batchsize, epochs, version=1):
    if version == 2:
        dataset = tf.contrib.data.TFRecordDataset('data/train2.tfrecord')
    else:
        dataset = tf.contrib.data.TFRecordDataset('data/train.tfrecord')

    def loadnp(ba):
        id = ba.decode('utf-8')
        return np.load(os.path.join(inputDir, 'training_data/feat', id + '.npy')).astype(np.float32)

    def loader(record):
        features = {
            # 'x': tf.FixedLenFeature([],tf.string)
            'caption': tf.FixedLenFeature([MAX_CAPTION_LEN], tf.int64),
            'caption_len': tf.FixedLenFeature([], tf.int64),
            'id': tf.FixedLenFeature([], tf.string)}
        parsed = tf.parse_single_example(record, features)
        x = tf.py_func(loadnp, [parsed['id']], tf.float32)
        x = tf.reshape(x, (80, 4096))
        return {
            'x': x,
            'caption_len': parsed['caption_len'],
            'id': parsed['id']}, parsed['caption']

    def eval_fn():
        ds = dataset.take(1000)
        ds = ds.map(loader)
        ds = ds.batch(batchsize)
        return ds.make_one_shot_iterator().get_next()

    def train_fn():
        ds = dataset.skip(1000)
        ds = ds.map(loader)
        ds = ds.shuffle(1000)
        ds = ds.repeat(epochs)
        ds = ds.batch(batchsize)
        return ds.make_one_shot_iterator().get_next()

    return train_fn, eval_fn


def testFromDir(inputDir, batchsize):
    ids = []
    dats = []
    dir = os.path.join(inputDir, 'testing_data/feat')
    for f in os.listdir(dir):
        if not f.endswith('.npy'):
            continue
        id = os.path.basename(f)[:-4]
        dat = np.load(os.path.join(dir, f)).astype(np.float32)
        ids.append(id)
        dats.append(dat)

    fn = tf.estimator.inputs.numpy_input_fn(
        x={
            'x':np.array(dats),
            'id': np.array(ids)},
        batch_size=batchsize,
        num_epochs=1,
        shuffle=False)
    return fn


def createEstimator(modeldir, model_fn, logsteps):
    params = None
    runConfig = tf.estimator.RunConfig()
    runConfig = runConfig.replace(
        log_step_count_steps=50,
        keep_checkpoint_max=2,
        save_checkpoints_steps=logsteps,
        save_summary_steps=logsteps)
    return tf.estimator.Estimator(
        model_fn=model_fn,
        params=params,
        config=runConfig,
        model_dir=modeldir)

def createExperiment(modeldir, model_fn, logsteps, inputDir, batchsize, epochs, version=1):
    train_fn, eval_fn = trainEvalFromTFRecord(inputDir, batchsize, epochs, version)
    estimator = createEstimator(modeldir, model_fn, logsteps)
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_fn,
        eval_input_fn=eval_fn,
        min_eval_frequency=1)
