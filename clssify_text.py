import datetime
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

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# filename = '../movie_review/nsmc/ratings_train.txt'
# datastore = pd.read_csv(filename, delimiter='\t')
# filename = '../movie_review/nsmc/ratings_test.txt'
# test_data = pd.read_csv(filename, delimiter='\t')

datastore = pd.read_csv('./train_spacing.csv')
test_data = pd.read_csv('./test_spacing.csv')

sentences = datastore.document.values
labels = datastore.label.values
ids = datastore.id.values

test_sentences = test_data.document.values
test_labels = test_data.label.values
test_ids = test_data.id.values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences.astype(str))
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences.astype(str))
sequences_test = tokenizer.texts_to_sequences(test_sentences.astype(str))
padded = pad_sequences(sequences, maxlen=int(np.quantile(np.array([len(seqs) for seqs in sequences]), 0.9)), padding='post', truncating='post')
padded_test = pad_sequences(sequences_test, maxlen=padded.shape[-1], padding='post', truncating='post')
padded = np.dstack([padded]*2)
padded_test = np.dstack([padded_test]*2)

vocab_size = len(word_index)
embedding_dim = 100
max_length = padded.shape[-2]
channel_num = 2

# padded = padded.reshape(len(padded),channel_num,-1)
# padded_test = padded_test.reshape(len(padded),channel_num,-1)
padded = np.transpose(padded, (0, 2,1))
padded_test = np.transpose(padded_test, (0, 2,1))

input1 = Input(shape=(channel_num, max_length))#일단 채널을 전달하고
x1 = Embedding(vocab_size+1, embedding_dim)(input1)

# x2 = tf.keras.layers.Reshape((max_length, embedding_dim,channel_num))(x1)
x2 = tf.transpose(x1, perm=[0,2,3,1])
x3 = []#여러개의 filter의 대표값을 저장할 공간

for i in range(2,6,1):
    #각 그램별 필터는 하나로 계산
    x31 = layers.Conv2D(filters=8, kernel_size=(i,embedding_dim), activation='relu', kernel_initializer='glorot_normal')(x2)
    x41 = MaxPool2D(pool_size=(x31.shape[1],1))(x31)
    x51 = Flatten()(x41)# 각 sample 별로 질문
    x3.append(x51)
merged = tf.concat(x3, axis=1)
x6 = Dense(64, kernel_initializer='glorot_normal')(merged)
x = layers.Dropout(0.5)(x6)
outputs = Dense(1, activation='sigmoid')(x)
model = Model(inputs=input1, outputs=outputs)

print(model.summary())

learning_rate = 0.001
batch_size = 100
lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, 
                                                        len(labels)/batch_size*5, 
                                                        decay_rate=0.5, 
                                                        staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)
model.compile(optimizer= optimizer, loss= binary_crossentropy, metrics=[binary_accuracy])
[]

MODEL_SAVE_FOLDER_PATH = '../trained_model'
model_file_path = f'{MODEL_SAVE_FOLDER_PATH}/review_cnn-{{epoch:d}}-{{val_loss:.5f}}-{{val_binary_accuracy:.5f}}.hdf5'
cb_model_check_point = ModelCheckpoint(filepath=model_file_path, monitor='val_binary_accuracy', verbose=1, save_best_only=True)
cb_early_stopping = EarlyStopping(monitor='val_loss', patience=6)

model.fit(padded, np.array(labels).reshape(-1,1), epochs=100, validation_split=0.2, batch_size=32
, callbacks=[cb_model_check_point, cb_early_stopping]
)


pred = model.predict(padded_test)
print(pred)
score = model.evaluate(padded_test,  np.array(test_labels).reshape(-1,1))
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




