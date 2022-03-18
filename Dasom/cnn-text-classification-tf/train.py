#! /usr/bin/env python

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences 
import fasttext
from konlpy.tag import Mecab
# Parameters
# ==================================================
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, Dense, GlobalMaxPooling2D
from keras.regularizers import L2
from keras.models import Model
from keras.optimizers import adam_v2
from keras.metrics import binary_accuracy
from keras.losses import binary_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

# Data loading params
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--train_data_file', default='/content/drive/MyDrive/data/naver_review/ratings_train.txt', type=str, help='')
parser.add_argument('--test_data_file', default='/content/drive/MyDrive/data/naver_review/ratings_test.txt', type=str, help='')
parser.add_argument('--model_dir', default='/content/drive/MyDrive/model/', type=str, help='')

# Model Hyperparameters
parser.add_argument('--embedding_dim', default=768, type=int, help='Dimensionality of character embedding (default: 128)')
parser.add_argument('--filter_sizes', default='3,4,5', type=str, help="Comma-separated filter sizes (default: '3,4,5')")
parser.add_argument('--num_filters', default=100, type=int, help='Number of filters per filter size (default: 100)')
parser.add_argument('--dropout_keep_prob', default=0.5, type=float, help='Dropout keep probability (default: 0.5)')
parser.add_argument('--l2_reg_lambda', default=0.5, type=float, help='L2 regularization lambda (default: 0.5)')
parser.add_argument('--non_static', default=True, type=bool, help='')
parser.add_argument('--model_type', default='transformer', type=str, help='')

# Training parameters
parser.add_argument('--batch_size', default=512, type=int, help='Batch Size (default: 64)')
parser.add_argument('--num_epochs', default=200, type=int, help='Number of training epochs (default: 200)')
parser.add_argument('--learning_rate', default=0.001, type=float, help='')
parser.add_argument('--decay_rate', default=0.5, type=float, help='')
# Misc Parameters

FLAGS = parser.parse_args()
print("\nParameters:")
for attr, value in FLAGS._get_kwargs():
    print("{}={}".format(attr.upper(), value))
print("")

def preprocess(embedding_model):
    # Data Preparation
    # ==================================================
    # Load data
    print("Loading data...")

    if FLAGS.model_type == 'transformer':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    elif FLAGS.model_type == 'fasttext':
        tokenizer = embedding_model

    def iterWithEmbedding(data_type):
        if data_type == 0:
            df = pd.read_table(FLAGS.train_data_file).dropna()
        elif data_type == 1:
            df = pd.read_table(FLAGS.test_data_file).dropna()
        for row in df.itertuples():
            if FLAGS.model_type == 'transformer':
                yield embedding_model.forward(torch.Tensor(tokenizer.encode(row.document)).to(torch.int64)).detach().numpy(), row.label
            elif FLAGS.model_type == 'fasttext':
                yield np.asarray([embedding_model[token] for token in embedding_model.f.tokenize(row.document)]).astype(np.float32), row.label
        del df
    train_ds = tf.data.Dataset.from_generator(iterWithEmbedding, args=[0]
        , output_signature=(
            tf.TensorSpec(shape=(None,FLAGS.embedding_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).shuffle(buffer_size=100, seed=42)
    val_ds = tf.data.Dataset.from_generator(iterWithEmbedding, args=[1]
        , output_signature=(
            tf.TensorSpec(shape=(None,FLAGS.embedding_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).shuffle(buffer_size=100, seed=42)
    return train_ds, val_ds#, test_ds

def train(train_ds, val_ds):
    # Training
    # ==================================================
    embedding_size, num_filters = FLAGS.embedding_dim, FLAGS.num_filters

    inputs = Input(shape=(None, embedding_size,1))# sequence length가 batch 별로 다름
    pooled_outputs = []
    filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
    for i, filter_size in enumerate(filter_sizes):
        conv = Conv2D(filters=num_filters,
            kernel_size=(filter_size, embedding_size),
            kernel_initializer = 'glorot_uniform',
            strides=(1,1), 
            padding='valid', 
            use_bias=True, 
            bias_initializer='zeros',
            activation='relu',
            data_format='channels_last',
        )(inputs)#TensorShape([None, None, 1, num_filters])
        pooled = GlobalMaxPooling2D(keepdims=True, data_format='channels_last')(conv)#TensorShape([None, 1, 1, num_filters])
        pooled_outputs.append(pooled)
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)# TensorShape([None, 1, 1, num_filters_total])
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])# TensorShape([None, num_filters_total])
    drop = Dropout(FLAGS.dropout_keep_prob)(h_pool_flat)# TensorShape([None, num_filters_total])
    outputs = Dense(
        1,
        activation='sigmoid', #이진 분류
        kernel_initializer='glorot_uniform', 
        kernel_regularizer=L2(l2=FLAGS.l2_reg_lambda),
        use_bias=True,
        bias_regularizer=L2(l2=FLAGS.l2_reg_lambda),
    )(drop)#TensorShape([None, 1])
    custom_model = Model(inputs=inputs, outputs =outputs)
    custom_model.summary()
    custom_model.layers[0].trainable = FLAGS.none_static

    learning_rate = FLAGS.learning_rate
    batch_size = FLAGS.batch_size#
    
    lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 
                                                        5000, 
                                                        decay_rate=FLAGS.decay_rate, 
                                                        staircase=True)
    optimizer = adam_v2.Adam(learning_rate=lr_decay)

    custom_model.compile(optimizer= optimizer, loss= binary_crossentropy, metrics=[binary_accuracy])
    model_file_path = f'{FLAGS.model_dir}/review_cnn-{{epoch:d}}-{{val_loss:.5f}}-{{val_binary_accuracy:.5f}}.hdf5'
    cb_model_check_point = ModelCheckpoint(filepath=model_file_path, monitor='val_binary_accuracy', verbose=1, save_best_only=True)
    cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)


    custom_model.fit(
        train_ds.padded_batch(batch_size), epochs=FLAGS.num_epochs#, batch_size=50
        , validation_data=val_ds.padded_batch(batch_size)
        , callbacks=[cb_model_check_point, cb_early_stopping]
    )

def main(argv=None):
    print("embedding model...")
    if FLAGS.model_type == 'transformer':
        embedding_model = AutoModel.from_pretrained("klue/roberta-base").embeddings.word_embeddings
    elif FLAGS.model_type == 'fasttext':
        embedding_model = fasttext.load_model('/content/drive/MyDrive/model/cc.ko.300.bin')# fast text로 진행

    train_ds, val_ds = preprocess(embedding_model)
    train(train_ds, val_ds)

if __name__ == '__main__':
    # app.run()
    main()