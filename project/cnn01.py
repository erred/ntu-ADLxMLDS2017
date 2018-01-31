import tensorflow as tf
import os
import sys

import iofn

BATCHSIZE = 256
EPOCHS = 200
ANCHORS = 5
CLASSES = 15
FINAL_DIMS = ANCHORS * (5 + CLASSES)
LOGSTEPS = 160
EVALSTEPS = 15
LEARNING_RATE = 0.001

VERSION = os.path.basename(os.path.splitext(sys.argv[0])[0])
MODELDIR = 'model/' + VERSION
tf.logging.set_verbosity(tf.logging.INFO)


def conv(x,
         filters,
         kernel=3,
         strides=1,
         padding='same',
         activation=tf.nn.leaky_relu,
         training=False):
    x = tf.layers.conv(
        x,
        filters=filters,
        kernel=kernel,
        strides=strides,
        padding=padding,
        activation=activation)
    x = tf.layers.batch_normalization(x, training=training)
    return x


def pool(x, kernel=2, strides=2, padding='same'):
    x = tf.layers.max_pooling2d(
        x, pool_size=kernel, strides=strides, padding=padding)
    return x


def model(features, labels, mode, params):
    x = features['x']

    training = False
    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True

    x = conv(x, 32, training=training)
    x = pool(x)
    x = conv(x, 64, training=training)
    x = pool(x)

    x = conv(x, 64, training=training)
    x = conv(x, 32, training=training)
    x = conv(x, 64, training=training)
    x = pool(x)

    x = conv(x, 128, training=training)
    x = conv(x, 64, training=training)
    x = conv(x, 128, training=training)
    x = pool(x)

    x = conv(x, 256, training=training)
    x = conv(x, 128, training=training)
    x = conv(x, 256, training=training)
    x = conv(x, 128, training=training)
    x = conv(x, 256, training=training)

    passthrough = tf.space_to_depth(x, 2)

    x = pool(x)

    x = conv(x, 512, training=training)
    x = conv(x, 256, training=training)
    x = conv(x, 512, training=training)
    x = conv(x, 256, training=training)
    x = conv(x, 512, training=training)

    x = conv(x, 512, training=training)
    x = conv(x, 512, training=training)

    x = tf.concat([x, passthrough], -1)
    x = conv(x, 512)

    x = tf.layers.conv2d(x, FINAL_DIMS)

    if mode == tf.estimator.ModeKeys.PREDICT:
        conf = tf.sigmoid(x[:, :, :, 0])
        xy = tf.sigmoid(x[:, :, :, 1:3])
        wh = tf.exp(x[:, :, :, 3:5])
        classif = tf.nn.softmax(x[:, :, :, 5:])

        predictions = {'x': x}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    else:
        conf = x[:, :, :, 0]
        xy = x[:, :, :, 1:3]
        wh = tf.exp(x[:, :, :, 3:5])
        classif = x[:, :, :, 5:]

        loss_conf = tf.losses.sigmoid_cross_entropy(labels[:, :, :, 0], conf)
        loss_reg_conf = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(conf), conf)
        loss_xy = tf.losses.mean_squared_error(labels[:, :, :, 1:3], xy)
        loss_wh = tf.losses.mean_squared_error(labels[:, :, :, 3:5], wh)
        loss_classif = tf.losses.softmax_cross_entropy(labels[:, :, :, 5:],
                                                       classif)

        loss = 5 * (
            loss_xy + loss_wh) + loss_conf + loss_classif + loss_reg_conf
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=LEARNING_RATE,
            optimizer=tf.train.AdamOptimizer())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)


if __name__ == "__main__":

    runConfig = tf.estimator.RunConfig()
    runConfig = runConfig.replace(
        log_step_count_steps=50,
        keep_checkpoint_max=2,
        save_checkpoints_steps=LOGSTEPS,
        save_summary_steps=LOGSTEPS)
    estimator = tf.estimator.Estimator(
        model_fn=model, params=None, config=runConfig, model_dir=MODELDIR)

    trainFn = iofn.withLabelFn(
        epochs=EPOCHS, batch_size=BATCHSIZE, filterfile='train.txt')
    evalFn = iofn.withLabelFn(
        epochs=EPOCHS, batch_size=BATCHSIZE, filterfile='valid.txt')

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=trainFn,
        eval_input_fn=evalFn,
        eval_steps=EVALSTEPS,
        eval_delay_secs=1,
        min_eval_frequency=1)
    experiment.train_and_evaluate()

    testFn = iofn.noLabelFn(batch_size=BATCHSIZE)
    preds = estimator.predict(input_fn=testFn)
