#! /usr/bin/env python

from jax import custom_gradient
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
# from tensorflow.contrib import learn
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences 
import fasttext
from konlpy.tag import Mecab
# Parameters
# ==================================================
from absl import flags, app
import argparse

# Data loading params
parser = argparse.ArgumentParser(description='Process some integers.')

# flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
parser.add_argument('--dev_sample_percentage', default=.1, type=float, help='')
# flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# parser.add_argument('--positive_data_file', default='/content/cnn_classifier/Dasom/cnn-text-classification-tf/data/rt-polaritydata/rt-polarity.pos', type=str, help='')
# flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
# parser.add_argument('--negative_data_file', default='/content/cnn_classifier/Dasom/cnn-text-classification-tf/data/rt-polaritydata/rt-polarity.neg', type=str, help='')
parser.add_argument('--train_data_file', default='/content/drive/MyDrive/data/naver_review/ratings_train.txt', type=str, help='')
parser.add_argument('--test_data_file', default='/content/drive/MyDrive/data/naver_review/ratings_test.txt', type=str, help='')

# Model Hyperparameters
# flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
# parser.add_argument('--embedding_dim', default=128, type=int, help='')
parser.add_argument('--embedding_dim', default=300, type=int, help='')
# flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
parser.add_argument('--filter_sizes', default='3,4,5', type=str, help='')
# flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# parser.add_argument('--num_filters', default=128, type=int, help='')
parser.add_argument('--num_filters', default=100, type=int, help='')
# flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
parser.add_argument('--dropout_keep_prob', default=0.5, type=float, help='')
# flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
parser.add_argument('--l2_reg_lambda', default=0.0, type=float, help='')

# Training parameters
# flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
parser.add_argument('--batch_size', default=64, type=int, help='')
# flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
parser.add_argument('--num_epochs', default=200, type=int, help='')
# flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
parser.add_argument('--evaluate_every', default=100, type=int, help='')
# flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
parser.add_argument('--checkpoint_every', default=100, type=int, help='')
# flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
parser.add_argument('--num_checkpoints', default=5, type=int, help='')
# Misc Parameters
# flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
parser.add_argument('--allow_soft_placement', default=True, type=bool, help='')
# flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
parser.add_argument('--log_device_placement', default=False, type=bool, help='')

# FLAGS = flags.FLAGS
FLAGS = parser.parse_args()
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def preprocess(embedding_model):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    # x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    train_df, test_df = data_helpers.load_data_and_labels(FLAGS.train_data_file, FLAGS.test_data_file)


    # Build vocabulary
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    # max_document_length = train_df.document.apply(lambda x: len(embedding_model.f.tokenize(x))).max()
    # vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    # vocab_processor = Tokenizer()
    # vocab_processor = learn.preprocessing.V   ocabularyProcessor(max_document_length)
    # x = np.array(list(vocab_processor.fit_transform(x_text)))
    # vocab_processor.fit_on_texts(x_text)
    # x= pad_sequences(vocab_processor.texts_to_sequences(x_text), maxlen=max_document_length, padding='post',truncating='post')
    m = Mecab()
    train_x = train_df.document.apply(lambda x: embedding_model.f.tokenize(x))# fasttext 모델로만 tokenizing을 하니 하나값으로 뭉치는 경우가 많음
    # train_x = train_df.document.apply(lambda x: embedding_model.f.tokenize(' '.join(m.morphs(x))))# fasttext 모델로만 tokenizing을 하니 하나값으로 뭉치는 경우가 많음
    train_y = train_df.label
    max_document_length = train_x.apply(len).max()
    def tmp(x):
        return x
    train_x.apply(tmp)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(train_y.shape[0]))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = y[shuffle_indices]
    x_shuffled = train_x.iloc[shuffle_indices]
    y_shuffled = train_y.iloc[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(train_y.shape[0]))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del train_x, train_y, x_shuffled, y_shuffled

    # fast text는 vacab size 불필요
    # print("Vocabulary Size: {:d}".format(len(vocab_processor.word_docs)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    # return x_train, y_train, vocab_processor, x_dev, y_dev
    return x_train, y_train, x_dev, y_dev, max_document_length


from keras.layers import Conv2D, MaxPool2D, Input, Dropout, Dense
from keras.regularizers import L2
from keras.models import Model
from keras.optimizers import adam_v2, backend
from keras.metrics import binary_accuracy, accuracy
from keras.losses import binary_crossentropy, CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

def train(x_train, y_train, vocab_processor, x_dev, y_dev, sequence_length):
    # Training
    # ==================================================
    embedding_size, num_filters = FLAGS.embedding_dim, FLAGS.num_filters
    assert embedding_size == vocab_processor.get_dimension()

    inputs = Input(shape=(sequence_length, embedding_size,1))
    pooled_outputs = []
    filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
    for i, filter_size in enumerate(filter_sizes):
        filter_shape = (filter_size, embedding_size, 1, num_filters)
        conv = Conv2D(filters=num_filters,
            # input_shape=(sequence_length,embedding_size,1),#batch_shape + (rows, cols, channels)
            # kernel_size=filter_shape,
            kernel_size=(filter_size, embedding_size),
            kernel_initializer = 'glorot_uniform',
            # strides=[1,1,1,1], 
            strides=(1,1), 
            padding='valid', 
            use_bias=True, 
            bias_initializer='zeros',
            activation='relu',
            data_format='channels_last',
        )(inputs)
        pooled = MaxPool2D(
            pool_size=(sequence_length - filter_size + 1, 1), 
            strides=(1,1), 
            padding='valid'
        )(conv)
        # if `data_format='channels_last'
        # 4+D tensor with shape: `batch_shape + (new_rows, new_cols, filters)` 
        pooled_outputs.append(pooled)
            # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    drop = Dropout(FLAGS.dropout_keep_prob)(h_pool_flat)
    outputs = Dense(
        2,
        activation='softmax', #이진 분류
        kernel_initializer='glorot_uniform', 
        kernel_regularizer=L2(l2=0.5),
        use_bias=True,
        bias_regularizer=L2(l2=0.5),
    )(drop)
    custom_model = Model(inputs=inputs, outputs =outputs)
    custom_model.summary()

    learning_rate = 0.001
    batch_size = 50
    
    lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 
                                                        len(labels)/batch_size*5, 
                                                        decay_rate=0.5, 
                                                        staircase=True)
    optimizer = adam_v2.Adam(learning_rate=lr_decay)

    custom_model.compile(optimizer= optimizer, loss= CategoricalCrossentropy, metrics=[accuracy])

    MODEL_SAVE_FOLDER_PATH = './models_prac'
    model_file_path = f'{MODEL_SAVE_FOLDER_PATH}/review_cnn-{{epoch:d}}-{{val_loss:.5f}}-{{val_accuracy:.5f}}.hdf5'
    cb_model_check_point = ModelCheckpoint(filepath=model_file_path, monitor='val_accuracy', verbose=1, save_best_only=True)
    cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # x_train, y_train, x_dev, y_dev

    custom_model.fit(padded, np.array(labels).reshape(-1,1), epochs=100, validation_split=0.2, batch_size=50
    , callbacks=[cb_model_check_point, cb_early_stopping]
    )

    del padded
    del labels

    pred = custom_model.predict(padded_test)
    score = custom_model.evaluate(pred,  np.array(test_labels).reshape(-1,1))
    print(score)



    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                # vocab_size=len(vocab_processor.word_docs),
                tokenizer=vocab_processor,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    print("embedding model...")
    embedding_model = fasttext.load_model('/content/drive/MyDrive/model/cc.ko.300.bin')# fast text로 진행

    # x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    x_train, y_train, x_dev, y_dev, max_document_length = preprocess(embedding_model)
    # train(x_train, y_train, vocab_processor, x_dev, y_dev)
    train(x_train, y_train, embedding_model, x_dev, y_dev, max_document_length)

if __name__ == '__main__':
    # app.run()
    main()