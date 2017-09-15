import tensorflow as tf
from mnist_reader import load_mnist

def cnn_model(features, labels, mode):
    with tf.name_scope("input_layer"):
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    with tf.name_scope("layer_1"):
        conv1 = tf.layers.conv2d(inputs=input_layer,
                                filters=8,
                                kernel_size=7,
                                padding="same",
                                activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=[2,2],
                                strides=2)

    with tf.name_scope("layer_2"):
        conv2 = tf.layers.conv2d(inputs=pool1,
                                filters=16,
                                kernel_size=5,
                                padding="same",
                                activation=tf.nn.relu)
        # pool2 = tf.layers.max_pooling2d(inputs=conv2,
        #                         pool_size=[2,2],
        #                         strides=2)

    with tf.name_scope("layer_3"):
        conv3 = tf.layers.conv2d(inputs=conv2,
                                filters=64,
                                kernel_size=3,
                                padding="same",
                                activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                pool_size=[2,2],
                                strides=2)

    with tf.name_scope("flatten_layer"):
        pool_flat = tf.reshape(pool3, [-1, 7 * 7 * 64])

    with tf.name_scope("dense_1"):
        dense1 = tf.layers.dense(inputs=pool_flat,
                                units=512,
                                activation=tf.nn.relu)
        dropout1 = tf.layers.dropout(inputs=dense1,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope("dense_2"):
        dense2 = tf.layers.dense(inputs=dropout1,
                                units=128,
                                activation=tf.nn.relu)
        dropout2 = tf.layers.dropout(inputs=dense2,
                                rate=0.4,
                                training=mode == tf.estimator.ModeKeys.TRAIN)

    with tf.name_scope("logits"):
        dense3 = tf.layers.dense(inputs=dropout2,units=10)


    predictions = {
        "classes": tf.argmax(input=dense3, axis=1),
        "probabilities": tf.nn.softmax(dense3, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=dense3)

  # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    train_data, train_labels = load_mnist('data/fashion', kind='train')
    eval_data, eval_labels = load_mnist('data/fashion', kind='t10k')
    predict_data, _ = load_mnist('predict', kind='t10k')

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model, model_dir="/tmp/mnist_convnet_model3")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)


    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=64,
        num_epochs=10,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    predict_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": predict_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    predict_results = mnist_classifier.evaluate(input_fn=predict_fn)
    fo = open("results.csv", 'w')
    listy = enumerate(predict_results)
    fo.write("id, label\n")
    for e in listy:
        fo.write("{},{}\n".format(e[0],e[1][classes]))
    fo.close()


if __name__ == "__main__":
    tf.app.run()
