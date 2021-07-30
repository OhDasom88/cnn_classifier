import datetime
from tensorflow import keras
import tensorflow
from tensorflow.python.keras.layers.core import Flatten

from tensorflow.python.keras.layers.pooling import MaxPool2D
start_time = datetime.datetime.now()
print(str(start_time))

from konlpy.tag import Mecab
import smart_open   
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Embedding, BatchNormalization
from tensorflow.keras import datasets, layers
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.losses import binary_crossentropy, CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import smart_open   
smart_open.open = smart_open.smart_open
from gensim.models import word2vec

model_type = 'CNN-static'
# model_type = None

# filename = './classifing_movie_review_kr/nsmc/ratings_train.txt'
# datastore = pd.read_csv(filename, delimiter='\t')
# filename = './classifing_movie_review_kr/nsmc/ratings_test.txt'
# test_data = pd.read_csv(filename, delimiter='\t')


datastore = pd.read_csv('./train_spacing.csv')
test_data = pd.read_csv('./test_spacing.csv')

sentences = datastore.document.values
labels = datastore.label.values
ids = datastore.id.values

test_sentences = test_data.document.values
test_labels = test_data.label.values
test_ids = test_data.id.values

tokenizer = Tokenizer(oov_token='oov')
tokenizer.fit_on_texts(sentences.astype(str))
word_index = tokenizer.word_index
vocabulary_inv = tokenizer.index_word# {v: k for k, v in tokenizer.word_index.items()}

sequences = tokenizer.texts_to_sequences(sentences.astype(str))
sequences_test = tokenizer.texts_to_sequences(test_sentences.astype(str))

padded = pad_sequences(sequences, maxlen=int(np.quantile(np.array([len(seqs) for seqs in sequences]), 0.9)), padding='post', truncating='post')
padded_test = pad_sequences(sequences_test, maxlen=padded.shape[-1], padding='post', truncating='post')


vocab_size = len(word_index)
embedding_dim = 200
max_length = padded.shape[-1]
channel_num = 2



if model_type in ["CNN-non-static", "CNN-static"]:
    # pre_trained
    embedding_model = word2vec.Word2Vec.load("./models_prac/ko.bin")
    vocabulary_inv.update({0:'pad'})
    embedding_weights = {key: embedding_model[word] if word in embedding_model else np.random.uniform(-0.25, 0.25, embedding_model.vector_size) for key, word in vocabulary_inv.items()}

    if model_type == "CNN-static":
        padded = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in padded])
        padded_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in padded_test])

elif model_type == "CNN-rand":
    embedding_weights = None



if model_type == "CNN-static":
    padded = np.stack([padded]*channel_num)
    padded_test = np.stack([padded_test]*channel_num)
    padded = np.transpose(padded, (1, 2,3,0))#(150000, 21, 200, 2)
    padded_test = np.transpose(padded_test, (1,2,3,0))#(50000, 21, 200, 2)

    input_shape = (padded.shape[1], padded.shape[2],padded.shape[3])
    #<KerasTensor: shape=(None, 2, 21, 200) dtype=float32 (created by layer 'input_1')>
    x2 = Input(shape=input_shape)
    model_input = x2
else:
    #(150000, 21, 2)
    padded = np.dstack([padded]*channel_num)
    padded_test = np.dstack([padded_test]*channel_num)

    padded = np.transpose(padded, (0, 2,1))
    padded_test = np.transpose(padded_test, (0, 2,1))

    model_input = Input(shape=(channel_num, max_length))

    x1 = Embedding(vocab_size+1, embedding_dim)(model_input)
    #<KerasTensor: shape=(None, 21, 200, 2) dtype=float32 (created by layer 'tf.compat.v1.transpose')>
    x2 = tf.transpose(x1, perm=[0,2,3,1])

x3 = []#여러개의 filter의 대표값을 저장할 공간

for i in range(2,6,1):
    #각 그램별 필터는 하나로 계산
    x31 = layers.Conv2D(filters=100, kernel_size=(i,embedding_dim), activation='relu', kernel_initializer='glorot_normal')(x2)
    x41 = MaxPool2D(pool_size=(x31.shape[1],1))(x31)
    x51 = Flatten()(x41)# 각 sample 별로 질문
    x3.append(x51)

merged = tf.concat(x3, axis=1)
x6 = Dense(64, kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(3))(merged)
x = layers.Dropout(0.5)(x6)
# outputs = Dense(2, activation='softmax')(x)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=model_input, outputs=outputs)

print(model.summary())

learning_rate = 0.001
batch_size = 50
lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 
                                                        len(labels)/batch_size*5, 
                                                        decay_rate=0.5, 
                                                        staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)

model.compile(optimizer= optimizer, loss= binary_crossentropy, metrics=[binary_accuracy])
# model.compile(optimizer= optimizer, loss= CategoricalCrossentropy(), metrics=[F1Score(num_classes=2, average='macro', threshold=None)])

MODEL_SAVE_FOLDER_PATH = './models_prac'
model_file_path = f'{MODEL_SAVE_FOLDER_PATH}/review_cnn-{{epoch:d}}-{{val_loss:.5f}}-{{val_binary_accuracy:.5f}}.hdf5'
cb_model_check_point = ModelCheckpoint(filepath=model_file_path, monitor='val_binary_accuracy', verbose=1, save_best_only=True)
cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.fit(padded, np.array(labels).reshape(-1,1), epochs=100, validation_split=0.2, batch_size=50
, callbacks=[cb_model_check_point, cb_early_stopping]
)

del padded
del labels

pred = model.predict(padded_test)
score = model.evaluate(pred,  np.array(test_labels).reshape(-1,1))
print(score)
# try:
#     model = Word2Vec.load("./word2vec.model")
# except:
#     try:
#         with open('./test_corpus.npy', 'rb') as f:
#             corpus= np.load(f, allow_pickle=True)
#         print(datetime.datetime.now()-start_time)
#     except:
#         filename = './classifing_movie_review_kr/nsmc/ratings_train.txt'
#         data = pd.read_csv(filename, delimiter='\t')
#         print(datetime.datetime.now()-start_time)
#         mecab = Mecab(r'C:\mecab\mecab-ko-dic')#210625 되는지 확인해보기
#         data['document'].fillna('', inplace=True)
#         corpus = [mecab.morphs(para) for para in data['document'][:100000]]
#         print(datetime.datetime.now()-start_time)

#         with open('./test_corpus.npy', 'wb') as f:
#             np.save(f, np.array(corpus, dtype=object))
#         print(datetime.datetime.now()-start_time)

#     model = Word2Vec(sentences=corpus, size=100, window=5, min_count=1, workers=4)
#     print(datetime.datetime.now()-start_time)

#     model.save("./word2vec.model")

# print(model)
# # vector = model.wv['금리']  # get numpy vector of a word
# # sims = model.wv.most_similar('금리', topn=10)  # get other similar words
# # print(vector)
# # print(sims)




