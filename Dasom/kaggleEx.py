import os
import numpy as np
from tqdm import tqdm
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Conv2D, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense
from keras.initializers import Constant
from keras import regularizers
from sklearn.metrics import accuracy_score


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

embeddings_index = {}
f = open(os.path.join('/content/drive/MyDrive/model/w2v', 'glove.6B.200d.txt'))
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

x_text, y = load_data_and_labels(
    '/content/cnn_classifier/Dasom/rt-polarity.pos'
    , '/content/cnn_classifier/Dasom/rt-polarity.neg'
)

max_features = 20000 # this is the number of words we care about
tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>')
tokenizer.fit_on_texts(x_text)
# this takes our sentences and replaces each word with an integer
X = tokenizer.texts_to_sequences(x_text)
# these sentences aren't that long so we may as well use the whole string
sequence_length = 52
# we then pad the sequences so they're all the same length (sequence_length)
X = pad_sequences(X, sequence_length)

# where there isn't a test set, Kim keeps back 10% of the data for testing, I'm going to do the same since we have an ok amount to play with
X_train, X_test, y_train, y_test = train_test_split(X, y[:,0], test_size=0.1)


word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


num_words = min(max_features, len(word_index)) + 1
print(num_words)

embedding_dim = 200

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector
        embedding_matrix[i] = np.random.randn(embedding_dim)

num_filters = 100

inputs_4 = Input(shape=(sequence_length,), dtype='int32')
embedding_layer_4 = Embedding(num_words,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=sequence_length,
                            trainable=True)(inputs_4)

reshape_4 = Reshape((sequence_length, embedding_dim, 1))(embedding_layer_4)

conv_0_4 = Conv2D(num_filters, kernel_size=(3, embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(3))(reshape_4)
conv_1_4 = Conv2D(num_filters, kernel_size=(4, embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(3))(reshape_4)
conv_2_4 = Conv2D(num_filters, kernel_size=(5, embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(3))(reshape_4)

maxpool_0_4 = MaxPool2D(pool_size=(sequence_length - 3 + 1, 1), strides=(1,1), padding='valid')(conv_0_4)
maxpool_1_4 = MaxPool2D(pool_size=(sequence_length - 4 + 1, 1), strides=(1,1), padding='valid')(conv_1_4)
maxpool_2_4 = MaxPool2D(pool_size=(sequence_length - 5 + 1, 1), strides=(1,1), padding='valid')(conv_2_4)

concatenated_tensor_4 = Concatenate(axis=1)([maxpool_0_4, maxpool_1_4, maxpool_2_4])
flatten_4 = Flatten()(concatenated_tensor_4)

dropout_4 = Dropout(0.5)(flatten_4)
# note the different activation
output_4 = Dense(units=1, activation='sigmoid')(dropout_4)


model_4 = Model(inputs=inputs_4, outputs=output_4)

# note we're using binary_crossentropy here instead of categorical
model_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_4.summary())

batch_size = 32
history_4 = model_4.fit(X_train, y_train, epochs=30, batch_size=batch_size, verbose=1, validation_split=0.2)

y_hat_4 = model_4.predict(X_test)

print(accuracy_score(y_test, list(map(lambda v: v > 0.5, y_hat_4))))







# 
df = pd.read_csv('/content/drive/MyDrive/data/kaggle_review/train.tsv', delimiter='\t')
df = df[['Phrase', 'Sentiment']]

df_0 = df[df['Sentiment'] == 0].sample(frac=1)
df_1 = df[df['Sentiment'] == 1].sample(frac=1)
df_2 = df[df['Sentiment'] == 2].sample(frac=1)
df_3 = df[df['Sentiment'] == 3].sample(frac=1)
df_4 = df[df['Sentiment'] == 4].sample(frac=1)

# we want a balanced set for training against - there are 7072 `0` examples
sample_size = 7072

# SST2 is the same as SST1, but with neutrals removed and binary labels
sst2_data = pd.concat([df_0.head(sample_size), df_1.head(sample_size), df_3.head(sample_size), df_4.head(sample_size)]).sample(frac=1)

def merge_sentiments(x):
    if x == 0 or x == 1:
        return 0
    else:
        return 1

sst2_data['Sentiment'] = sst2_data['Sentiment'].apply(merge_sentiments)

max_features = 20000 # this is the number of words we care about
# these sentences aren't that long so we may as well use the whole string
sequence_length = 52

sst2_tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>')
sst2_tokenizer.fit_on_texts(sst2_data['Phrase'].values)

sst2_X = sst2_tokenizer.texts_to_sequences(sst2_data['Phrase'].values)
sst2_X = pad_sequences(sst2_X, sequence_length)

sst2_y = sst2_data['Sentiment'].values

sst2_X_train, sst2_X_test, sst2_y_train, sst2_y_test = train_test_split(sst2_X, sst2_y, test_size=0.1)

print('Found %s word vectors.' % len(embeddings_index))

word_index = sst2_tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


num_words = min(max_features, len(word_index)) + 1
print(num_words)

embedding_dim = 200

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((num_words, embedding_dim))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector
        embedding_matrix[i] = np.random.randn(embedding_dim)

num_filters = 100

inputs_4 = Input(shape=(sequence_length,), dtype='int32')
embedding_layer_4 = Embedding(num_words,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=sequence_length,
                            trainable=True)(inputs_4)

reshape_4 = Reshape((sequence_length, embedding_dim, 1))(embedding_layer_4)

conv_0_4 = Conv2D(num_filters, kernel_size=(3, embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(3))(reshape_4)
conv_1_4 = Conv2D(num_filters, kernel_size=(4, embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(3))(reshape_4)
conv_2_4 = Conv2D(num_filters, kernel_size=(5, embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(3))(reshape_4)

maxpool_0_4 = MaxPool2D(pool_size=(sequence_length - 3 + 1, 1), strides=(1,1), padding='valid')(conv_0_4)
maxpool_1_4 = MaxPool2D(pool_size=(sequence_length - 4 + 1, 1), strides=(1,1), padding='valid')(conv_1_4)
maxpool_2_4 = MaxPool2D(pool_size=(sequence_length - 5 + 1, 1), strides=(1,1), padding='valid')(conv_2_4)

concatenated_tensor_4 = Concatenate(axis=1)([maxpool_0_4, maxpool_1_4, maxpool_2_4])
flatten_4 = Flatten()(concatenated_tensor_4)

dropout_4 = Dropout(0.5)(flatten_4)
# note the different activation
output_4 = Dense(units=1, activation='sigmoid')(dropout_4)

model_4 = Model(inputs=inputs_4, outputs=output_4)

# note we're using binary_crossentropy here instead of categorical
model_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_4.summary())

batch_size = 32
history_4 = model_4.fit(sst2_X_train, sst2_y_train, epochs=30, batch_size=batch_size, verbose=1, validation_split=0.2)
y_hat_4 = model_4.predict(sst2_X_test)

print(accuracy_score(sst2_y_test, list(map(lambda v: v > 0.5, y_hat_4))))



