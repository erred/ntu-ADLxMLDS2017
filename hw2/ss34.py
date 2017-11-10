import sys

import numpy as np
import tensorflow as tf

import iofn

# =================== Hyperparams
VERSION = 'ss34'
BATCHSIZE = 128
EPOCHS = 12
EMBED_DIMS = 20
LEARNING_RATE = 0.001
SHARED_UNITS = 1024
# =================== Constants
# TRAIN_SIZE = 24232
TRAIN_SIZE = 21939
NUM_CLASSES = 6016 + 2
MAX_OUTPUT_LEN = 45
GO_SYMBOL = 1
END_SYMBOL = 0
MODELDIR = 'model/' + VERSION + '/'
EPOCH_STEPS = TRAIN_SIZE / BATCHSIZE
tf.logging.set_verbosity(tf.logging.INFO)
# =================== Helper function

# =================== Decoder
def train_decoder(cell, encoder_final_state, projection, embedded_labels, embedded_labels_len):
    # helper = tf.contrib.seq2seq.TrainingHelper(
    #     inputs=embedded_labels,
    #     sequence_length=embedded_labels_len)

    # prob = tf.train.polynomial_decay(1.0, tf.train.get_global_step(), 100000, 0)
    # prob = tf.train.natural_exp_decay(1.0, tf.train.get_global_step(), 10000, 0.5)
    prob = tf.train.inverse_time_decay(1.0, tf.train.get_global_step(), 100000, 0.5)
    # prob = tf.train.exponential_decay(1.0, tf.train.get_global_step(), 100000, 0.96)
    embeddings = tf.get_variable('embeddings', shape=(NUM_CLASSES, EMBED_DIMS))
    helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        inputs=embedded_labels,
        sequence_length=embedded_labels_len,
        embedding=embeddings,
        sampling_probability=prob)
    return basic_decoder(cell, helper, encoder_final_state, projection)

def predict_decoder(cell, encoder_final_state, projection, batchsize):
    embeddings = tf.get_variable('embeddings', shape=(NUM_CLASSES, EMBED_DIMS))
    # Sample or Argmax
    # helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
    #     embedding=embeddings,
    #     start_tokens=tf.tile([GO_SYMBOL], [batchsize]),
    #     end_token=END_SYMBOL)
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding=embeddings,
        start_tokens=tf.tile([GO_SYMBOL], [batchsize]),
        end_token=END_SYMBOL)
    return basic_decoder(cell, helper, encoder_final_state, projection)

def basic_decoder(cell, helper, encoder_final_state, projection):
    return tf.contrib.seq2seq.BasicDecoder(
        cell=cell,
        helper=helper,
        initial_state=encoder_final_state,
        output_layer=projection)

# def beam_decoder(cell, encoder_final_state, projection, beam_width, batchsize):
#     embeddings = tf.get_variable('embeddings', shape=(NUM_CLASSES, EMBED_DIMS))
#     tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(
#         encoder_final_state, multiplier=beam_width)
#     # decoder_initial_state = cell.zero_state(
#     #     dtype=tf.float32, batch_size=batchsize * beam_width)
#     # decoder_initial_state = decoder_initial_state.clone(
#     #     cell_state=tiled_encoder_final_state)
#     return tf.contrib.seq2seq.BeamSearchDecoder(
#         cell=cell,
#         embedding=embeddings,
#         start_tokens=tf.tile([GO_SYMBOL], [batchsize]),
#         end_token=END_SYMBOL,
#         initial_state=tiled_encoder_final_state,
#         beam_width=beam_width,
#         output_layer=projection)


# =================== Define Model
def model_fn(features, labels, mode):

    # =================== Inputs
    ids = features['id']
    inputs = features['x']
    batch_size = tf.shape(inputs)[0]
    if mode != tf.estimator.ModeKeys.PREDICT:
        labels = labels
        embedded_labels = tf.contrib.layers.embed_sequence(
            ids=tf.pad(labels[:,:-1],[[0,0],[1,0]], constant_values=1),
            vocab_size=NUM_CLASSES,
            embed_dim=EMBED_DIMS)
        labels_len = tf.cast(features['caption_len'], tf.int32)


    # =================== Encoder
    encoder_cell = tf.contrib.rnn.LSTMCell(
        num_units=SHARED_UNITS)
    # encoder_cell = tf.contrib.rnn.GRUCell(
    #     num_units=SHARED_UNITS)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        cell=encoder_cell,
        inputs=inputs,
        dtype=tf.float32)

    # ==================== Decoder
    projection = tf.layers.Dense(
        units=NUM_CLASSES,
        activation=tf.nn.relu)
    decoder_cell = tf.contrib.rnn.LSTMCell(
        num_units=SHARED_UNITS)
    # decoder_cell = tf.contrib.rnn.GRUCell(
    #     num_units=SHARED_UNITS)

    attn_mech = tf.contrib.seq2seq.LuongAttention(
        num_units=SHARED_UNITS,
        memory=encoder_output,
        scale=True)
    attn_mech2 = tf.contrib.seq2seq.BahdanauAttention(
        num_units=SHARED_UNITS,
        memory=encoder_output,
        normalize=True)
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell=decoder_cell,
        attention_mechanism=[attn_mech, attn_mech2],
        attention_layer_size=[1024, 1024])
    decoder_initial_state = decoder_cell.zero_state(
        dtype=tf.float32, batch_size=batch_size)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=encoder_state)

    # decoder = beam_decoder(decoder_cell, encoder_state, projection, 3, batch_size)

    if mode == tf.estimator.ModeKeys.PREDICT:
        decoder = predict_decoder(decoder_cell, decoder_initial_state, projection, batch_size)
    else:
        decoder = train_decoder(
            decoder_cell,
            decoder_initial_state,
            projection,
            embedded_labels,
            labels_len)


    decode_output, _, output_len = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        impute_finished = True,
        maximum_iterations=MAX_OUTPUT_LEN)

    output = decode_output.rnn_output

    def padOutput(elems):
        pad_amount = MAX_OUTPUT_LEN - tf.shape(elems[0])[0]
        padded = tf.pad(elems[0], [[0, pad_amount], [0, 0]])
        padded = tf.reshape(padded,(MAX_OUTPUT_LEN, NUM_CLASSES))
        return padded, elems[1]
    output, _ = tf.map_fn(
        fn=padOutput,
        elems=(output, output_len))

    output = tf.reshape(output, [-1, MAX_OUTPUT_LEN, NUM_CLASSES])


    # =================== Model Ends Here
    if mode == tf.estimator.ModeKeys.PREDICT:
        return modelSpec(mode, output, ids=ids, output_len=output_len)
    return modelSpec(mode, output, labels=labels, output_len=output_len)




# =================== Make TF run
def modelSpec(mode, output, ids=None, labels=None, output_len=None):
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'prediction': tf.argmax(output, 2),
            'id': ids,
            'len': output_len}
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)
    else:
        mask = tf.cast(tf.sequence_mask(output_len, MAX_OUTPUT_LEN), tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(
            logits=output,
            targets=labels,
            weights=mask)
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=LEARNING_RATE)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=LEARNING_RATE,
            optimizer=optimizer)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == 'train':
        # inputDir = sys.argv[2]
        inputDir = 'data'
        exp = iofn.createExperiment(
            MODELDIR, model_fn, EPOCH_STEPS, inputDir, BATCHSIZE, EPOCHS)
        exp.train()
    else:
        inputDir = sys.argv[2]
        outputFile = sys.argv[3]

        test_fn = iofn.testFromDir(inputDir, BATCHSIZE)
        est = iofn.createEstimator(MODELDIR, model_fn, EPOCH_STEPS)
        predictions = est.predict(input_fn=test_fn)
        preds = [
            [
                p['id'].decode('utf-8'),
                iofn.decodeCaption(p['prediction'][:p['len']])
            ]
            for p in predictions]

        if mode == 'test-special':
            special = ['klteYv1Uv9A_27_33.avi', '5YJaS2Eswg0_22_26.avi', 'UbmZAe5u5FI_132_141.avi', 'JntMAcTlOF0_50_70.avi', 'tJHUH9tpqPg_113_118.avi']
            preds = [p for p in preds if p[0] in special]

        print(preds)
        iofn.saveOutput(preds, outputFile)

        if mode != 'test-special':
            peerOutputFile = sys.argv[4]

            test_fn = iofn.testFromDir(inputDir, BATCHSIZE, 'peer')
            # est = iofn.createEstimator(MODELDIR, model_fn, EPOCH_STEPS)
            predictions = est.predict(input_fn=test_fn)
            preds = [
                [
                    p['id'].decode('utf-8'),
                    iofn.decodeCaption(p['prediction'][:p['len']])
                ]
                for p in predictions]
            print(preds)
            iofn.saveOutput(preds, peerOutputFile)
