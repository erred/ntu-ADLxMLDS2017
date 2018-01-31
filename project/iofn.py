import tensorflow as tf
import os
import csv

FINDINGS = {
    'No Finding': 0,
    'Atelectasis': 1,
    'Cardiomegaly': 2,
    'Consolidation': 3,
    'Edema': 4,
    'Effusion': 5,
    'Emphysema': 6,
    'Fibrosis': 7,
    'Hernia': 8,
    'Infiltration': 9,
    'Mass': 10,
    'Nodule': 11,
    'Pleural_Thickening': 12,
    'Pneumonia': 13,
    'Pneumothorax': 14
}


def read_bbox(fname='data/BBox_List_2017.csv'):
    bboxes = []
    with open(fname) as f:
        re = csv.reader(f)
        next(re)
        for r in re:
            # Image Index,Finding Label,Bbox [x,y,w,h],,,
            bboxes.append(r)
    return bboxes


def withLabelFn(epochs=1,
                batch_size=128,
                basedir='data',
                filterfile='train.txt',
                labelfile='Data_Entry_2017_v2.csv',
                shuffle_buffer=10000,
                parallel=3):

    with open(os.path.join(basedir, filterfile)) as f:
        filterset = set([x.replace('\n', '') for x in f.readlines()])

    def g():
        with open(os.path.join(basedir, labelfile)) as f:
            re = csv.reader(f)
            next(re)
            for r in re:
                if r[0] not in filterset:
                    continue
                # Image Index,Finding Labels,Follow-up #,Patient ID,
                # Patient Age,Patient Gender,View Position,
                # OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],
                f_oh = [0] * 15
                for f in r[1].split('|'):
                    f_oh[FINDINGS[f]] = 1
                yield os.path.join(basedir, 'images', r[0]), f_oh

    g_types = (tf.string, tf.int64)
    g_shape = (tf.TensorShape([]), tf.TensorShape([15]))

    ds = tf.data.Dataset.from_generator(g, g_types, g_shape)
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.repeat(epochs)
    ds = ds.map(lambda x, y: (tf.image.decode_png(tf.read_file(x)), y))
    ds = ds.map(lambda x, y: ({'x': x}, y))
    ds = ds.batch(batch_size)
    return lambda: ds.make_one_shot_iterator().get_next()


def noLabelFn(batch_size=128,
              basedir='data',
              filterfile='test.txt',
              parallel=3):

    ds = tf.data.TextLineDataset(os.path.join(basedir, filterfile))
    ds = ds.map(lambda x: tf.image.decode_png(tf.read_file(x)))
    ds = ds.map(lambda x: {'x': x})
    ds = ds.batch(batch_size)
    return lambda: ds.make_one_shot_iterator().get_next()
