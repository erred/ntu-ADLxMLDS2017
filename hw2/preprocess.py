import json
import os
import pickle

import tensorflow as tf
from keras.preprocessing.text import text_to_word_sequence

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

def createTranslationMap(inputDir):
    """utility function
    reads captions and creates encoding maps
    """
    with open(os.path.join(inputDir, 'training_label.json'), 'rb') as fo:
        v_tr = json.load(fo)

    m = []

    s = set()
    for v in v_tr:
        for sen in v["caption"]:
            low = text_to_word_sequence(sen)
            m.append(len(low))
            for w in low:
                s.add(w)

    w2e = {' ': 0}
    e2w = {0: ' ', 1: ' '}
    for i, w in enumerate(s):
        w2e[w] = i+1
        e2w[i+1] = w

    print("reserve 1 for START, 0 for END")
    print("longest senctence (no symbols): ", max(m))
    print("total words (no symbols): ", len(s))
    return w2e, e2w


def saveTranslationMap():
    w2e, e2w = createTranslationMap('data')
    with open('data/w2e.pkl', 'wb') as fo:
        pickle.dump(w2e, fo)
    with open('data/e2w.pkl', 'wb') as fo:
        pickle.dump(e2w, fo)



def floatFeature(v):
    return tf.train.Feature(float_list=tf.train.FloatList(value=v))
def int64Feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))
def bytesFeature(v):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=v))


# encoded + [EOS]
# add 1 to len
def tfrecordFromTrain():
    inputDir = 'data'
    outputFile = 'data/train.tfrecord'
    writer = tf.python_io.TFRecordWriter(outputFile)

    with open(os.path.join(inputDir, 'training_label.json')) as fo:
        capj = json.load(fo)
    for rec in capj:
        # x = np.load(os.path.join(inputDir, 'training_data/feat', rec['id'] + '.npy'))
        # captions = [[1] + encodeCaption(s) + [0] for s in rec['caption']]
        # caption_lens = [len(c) for c in captions]
        # captions = [c + [0] * (MAX_CAPTION_LEN - len(c)) for c in captions]
        for cap in rec['caption']:
            c = encodeCaption(cap) + [0]
            len_c = len(c) + 1
            c = c + [0] * (MAX_CAPTION_LEN - len(c))

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        # 'x': bytesFeature(x.tostring()),
                        'caption': int64Feature(c),
                        'caption_len': int64Feature([len_c]),
                        'id': bytesFeature([rec['id'].encode('utf-8')])
                    }))
            writer.write(example.SerializeToString())
