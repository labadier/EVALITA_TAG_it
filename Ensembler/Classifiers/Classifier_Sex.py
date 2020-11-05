# In[]
import sys
sys.path.append('../../')
from toolsIT import *
import tensorflow as tf
import keras as K
import numpy as np
import keras_self_attention
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_confusion_matrix
import sklearn.metrics
import seaborn
import os
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute Weights for Tweet Representation')
    parser.add_argument('-d', metavar='-data_path', required=True, help='The Path of the data')
    args = parser.parse_args()
    DATA_PATH = args.d

    EMBD_PATH = "../../data/wiki-it.vec"
    embeddings_matrix, dicc = read_embedding(EMBD_PATH)
    encode_seqs, _, index = load_for_age(embeddings_matrix, dicc, DATA_PATH, 'sex', 'test')

    sentence_encode = np.load('../../data/profiles_sentence_encode.npy')
    Bert_rep = np.load('../../data/Bert_encode.npy')
    featuresmine = np.load('../../data/profiles_features.npy')
    
    featuresmine = np.log10(featuresmine+0.00001)
        
    features_itfidf = np.load('../../data/features_IG_itfidf_gender.npy')
    features = np.concatenate([features_itfidf, featuresmine], axis= 1)

    print('Examples Shape:', encode_seqs.shape)
    embb_lay = K.layers.Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1],
                           embeddings_initializer=K.initializers.Constant(embeddings_matrix), input_length=400, trainable=False, name='embd')

    def Model(inshape, inshape_sentences, feashape, besha):
        I = K.layers.Input(inshape, name='input_seq')
        X = embb_lay(I)
            
        concat_seq = keras_self_attention.ScaledDotProductAttention(name='XprodATT2_seq')(X)
        concat_seq = K.layers.Dropout(rate=0.3, name='dp_seq')(concat_seq)
        
        rnncell_seq = K.layers.Bidirectional(K.layers.LSTM(units=32, name='lstmcell_seq', return_sequences=True))(concat_seq)
        rnncell_seq = K.layers.Dropout(rate=0.2, name='lstmdp_seq')(rnncell_seq)
        rnncell_seq = K.layers.LSTM(units=32, name='lstmcell2_seq')(rnncell_seq)

        S = K.layers.Input(inshape_sentences, name='input_sentences')
            
        concat_sentences = keras_self_attention.ScaledDotProductAttention(name='XprodATT2_sentences')(S)
        concat_sentences = K.layers.Dropout(rate=0.3, name='dp_sentences')(concat_sentences)
        
        rnncell_sentences = K.layers.Bidirectional(K.layers.LSTM(units=32, name='lstmcell_sentences', return_sequences=True))(concat_sentences)
        rnncell_sentences = K.layers.Dropout(rate=0.2, name='lstmdp_sentences')(rnncell_sentences)
        rnncell_sentences = K.layers.LSTM(units=32, name='lstmcell2_sentences')(rnncell_sentences)
        
        F = K.layers.Input(feashape)
        B = K.layers.Input(besha)
        f = K.layers.Dense(units=64, activation=None, name='densecompaq', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.04))(F)
        f = K.layers.LeakyReLU(alpha=0.3)(f)
    #     f = K.layers.Dropout(rate=0.3, name='dpfeat')(f)

        flatt = K.layers.concatenate([rnncell_seq, rnncell_sentences,  B])
        flatt = K.layers.Dense(units=64, name='dense', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.04))(flatt)
        flatt = K.layers.LeakyReLU(alpha=0.3)(flatt)
        
        Y = K.layers.Dense(2, activation='softmax', name='densex')(flatt)
        return K.Model(inputs=[I, S, F, B], outputs=Y)

    Classifier = Model(encode_seqs[0].shape, sentence_encode[0].shape, features[0].shape, Bert_rep[0].shape)
    Classifier.compile(optimizer=K.optimizers.Adam(lr=2e-3),loss=K.losses.CategoricalCrossentropy(), metrics=['acc'])
    Classifier.load_weights('../sex_model_[B|itfidf_Mine].h5', by_name=True)

    with open('../../output/output_sex.txt', 'w') as file:
        Y = Classifier.predict([encode_seqs, sentence_encode, features, Bert_rep])
        Y = np.argmax(Y, axis=1)
        for i in range(len(Y)):
            file.write(index[i] + ' ' + SEX_CLASSES[Y[i]] + '\n')
    os.system('clear')
    print('Gender Prediction Done !!!')