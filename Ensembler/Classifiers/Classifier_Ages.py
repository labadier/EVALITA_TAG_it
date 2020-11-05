# %%

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
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics
import seaborn
import os
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Precompute Weights for Tweet Representation')
    parser.add_argument('-d', metavar='-data_path', required=True, help='The Path of the data')
    args = parser.parse_args()
    DATA_PATH = args.d

    EMBD_PATH = "../../data/wiki-it.vec" 

    embeddings_matrix, dicc = read_embedding(EMBD_PATH)
    encode_seqs, _, index = load_for_age(embeddings_matrix, dicc, DATA_PATH, 'age', 'test')

    sentence_encode = np.load('../../data/profiles_sentence_encode.npy')
    Bert_feat = np.load('../../data/Bert_encode.npy')
    featuresmine = np.load('../../data/profiles_features.npy')
    
    featuresmine = np.log10(featuresmine+0.00001)
        
    features_itfidf = np.load('../../data/features_IG_itfidf_age.npy')
    features = np.concatenate([features_itfidf, featuresmine], axis= 1)
    embb_lay = K.layers.Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1],
                           embeddings_initializer=K.initializers.Constant(embeddings_matrix), input_length=400, trainable=False, name='embd')

    def Model(inshape, shapesentences, feashape, besha):
        I = K.layers.Input(inshape, name='input_seq')
        X = embb_lay(I)
        
        concat_seq = keras_self_attention.ScaledDotProductAttention(name='XprodATT2_seq')(X)
        concat_seq = K.layers.Dropout(rate=0.3, name='dp_seq')(concat_seq)
        
        rnncell_seq = K.layers.Bidirectional(K.layers.LSTM(units=64, name='lstmcell_seq', return_sequences=True))(concat_seq)
        rnncell_seq = K.layers.Dropout(rate=0.2, name='lstmdp_seq')(rnncell_seq)
        rnncell_seq = K.layers.LSTM(units=64, name='lstmcell2_seq')(rnncell_seq)

        S = K.layers.Input(shapesentences, name='input_sentences')
        
        concat_sentences = keras_self_attention.ScaledDotProductAttention(name='XprodATT2_sentences')(S)
        concat_sentences = K.layers.Dropout(rate=0.3, name='dp_sentences')(concat_sentences)
        
        rnncell_sentences = K.layers.Bidirectional(K.layers.LSTM(units=64, name='lstmcell_sentences', return_sequences=True))(concat_sentences)
        rnncell_sentences = K.layers.Dropout(rate=0.2, name='lstmdp_sentences')(rnncell_sentences)
        rnncell_sentences = K.layers.LSTM(units=64, name='lstmcell2_sentences')(rnncell_sentences)

        F = K.layers.Input(feashape, name='input_festures')
        B = K.layers.Input(besha, name='Bert')

        f = K.layers.Dense(units=64, activation=None, name='densecompaq', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02))(F)
        f = K.layers.LeakyReLU(alpha=0.3)(f)
        
        flatt = K.layers.concatenate([rnncell_seq, rnncell_sentences, B], name='concatenate')
        flatt = K.layers.Dense(units=64, activation=None, name='preout')(flatt)
        flatt = K.layers.LeakyReLU(alpha=0.2)(flatt)

        Y = K.layers.Dense(5, activation='softmax', name='densex')(flatt)
        return K.Model(inputs=[I, S, F, B], outputs=Y)

    
    Classifier = Model(encode_seqs[0].shape, sentence_encode[0].shape, features[0].shape, Bert_feat[0].shape)
    Classifier.compile(optimizer=K.optimizers.Adam(lr=2e-3, decay = 5e-5), loss=K.losses.CategoricalCrossentropy(), metrics=['acc'])
    Classifier.load_weights('../ages_model_[B|itfidf_Mine].h5', by_name=True)
    
    with open('../../output/output_age.txt', 'w') as file:
        Y = Classifier.predict([encode_seqs, sentence_encode, features, Bert_feat])
        Y = np.argmax(Y, axis=1)
        for i in range(len(Y)):
            file.write(index[i] + ' ' + AGE_CLASSES[Y[i]] + '\n')
    os.system('clear')
    print('Age Prediction Done !!!')
