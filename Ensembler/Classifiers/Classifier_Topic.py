import sys
sys.path.append('../../')
from toolsIT import *
import tensorflow as tf
import keras as K
import numpy as np
import keras_self_attention
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
import json
from sklearn.metrics import plot_confusion_matrix
import sklearn.metrics
import seaborn
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute Weights for Tweet Representation')
    parser.add_argument('-d', metavar='-data_path', required=True, help='The Path of the data')
    args = parser.parse_args()
    DATA_PATH = args.d
    EMBD_PATH = "../../data/wiki-it.vec"

    stop_words = []
    itsw =  open('../../data/stopwords-it.txt', mode='rU')
    for wordors in itsw:
        stop_words.append(wordors[:-1])

    embeddings_matrix, dicc = read_embedding(EMBD_PATH)
    encode_seqs, _, index = lematized_for_topic(embeddings_matrix, dicc, DATA_PATH, 'test', stop_words)

    sentence_encode = np.load('../../data/profiles_sentence_encode.npy')
    Bert_feat = np.load('../../data/Bert_encode.npy')
    featuresmine = np.load('../../data/profiles_features.npy')
    
    featuresmine = np.log10(featuresmine+0.00001)
        
    features_itfidf = np.load('../../data/features_IG_itfidf_topic.npy')
    features = np.concatenate([features_itfidf, featuresmine], axis= 1)
    embb_lay = K.layers.Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1],
                           embeddings_initializer=K.initializers.Constant(embeddings_matrix), input_length=400, trainable=False, name='embd')

    def Model(inshape, inshape_setence, feashape, besha):
        
        I = K.layers.Input(inshape, name='input_seq')
        X = embb_lay(I)
        concat_seq = keras_self_attention.ScaledDotProductAttention(name='XprodATT_seq')(X)
        concat_seq = K.layers.Dropout(rate=0.3, name='dp0_seq')(concat_seq)

        rnncell_seq = K.layers.Bidirectional(K.layers.LSTM(units=32, name='lstmcell_', return_sequences=True))(concat_seq)
        rnncell_seq = K.layers.Dropout(rate=0.2, name='lstmdp_seq')(rnncell_seq)
        rnncell_seq = K.layers.LSTM(units=32, name='lstmcell2_seq')(rnncell_seq)

        S = K.layers.Input(inshape_setence, name='input_seneteces')
        concat_seneteces = keras_self_attention.ScaledDotProductAttention(name='XprodATT_seneteces')(S)
        concat_seneteces = K.layers.Dropout(rate=0.3, name='dp0_seneteces')(concat_seneteces)

        rnncell_seneteces = K.layers.Bidirectional(K.layers.LSTM(units=32, name='lstmcell_seneteces', return_sequences=True))(concat_seneteces)
        rnncell_seneteces = K.layers.Dropout(rate=0.2, name='lstmdp_seneteces')(rnncell_seneteces)
        rnncell_seneteces = K.layers.LSTM(units=32, name='lstmcell2_seneteces')(rnncell_seneteces)

        F = K.layers.Input(feashape)
        B = K.layers.Input(besha)
        f = K.layers.Dense(units=64, activation=None, name='densecompaq', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.05))(F)
        f = K.layers.LeakyReLU(alpha=0.005)(f)

        flatt = K.layers.concatenate([rnncell_seq, rnncell_seneteces, B])
        flatt = K.layers.Dense(units=64, name='dense')(flatt)
        flatt = K.layers.LeakyReLU(alpha=0.1)(flatt)

        Y = K.layers.Dense(11, activation='softmax', name='densex_')(flatt)
                
        return K.Model(inputs=[I, S, F, B], outputs=Y)

    Classifier = Model(encode_seqs[0].shape, sentence_encode[0].shape, features[0].shape, Bert_feat[0].shape)
    Classifier.compile(optimizer=K.optimizers.Adam(lr=3e-3, decay = 1e-5), loss=K.losses.CategoricalCrossentropy(), metrics=['acc'])
    Classifier.load_weights('../topic_model_[B|itfidf_Mine].h5', by_name=True)

    with open('../../output/output_topic.txt', 'w') as file:
        Y = Classifier.predict([encode_seqs, sentence_encode, features, Bert_feat])
        Y = np.argmax(Y, axis=1)
        print(len(Y))
        for i in range(len(Y)):
            print(index[i] , TOPIC_CLASSES[Y[i]] )
            file.write(index[i] + ' ' + TOPIC_CLASSES[Y[i]] + '\n')
    os.system('clear')
    print('Topic Prediction Done !!!')