# In[]
import sys
sys.path.append('../')
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
import json
import argparse
from keras import backend as bk

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Precompute Weights for Tweet Representation')
    parser.add_argument('-md', metavar='-model', required=True, help='Model Composition')
    parser.add_argument('-dp', metavar='-datapath', required=True, help='Trainning Data Path')
    args = parser.parse_args()
    model = args.md
        
    EMBD_PATH = "../data/wiki-it.vec"
    DATA_PATH = args.dp

    embeddings_matrix, dicc = read_embedding(EMBD_PATH)
    encode_seqs, labels, _ = load_for_age(embeddings_matrix, dicc, DATA_PATH, 'sex', 'train')

    # np.save('labls_sex.npy', labels)
    # np.save('seqs_sex.npy', encode_seqs)

    sentences_encode = np.load('../data/profiles_sentence_encode.npy')
    # labels = np.load('labls_sex.npy')
    # encode_seqs = np.load('seqs_sex.npy')
    # embeddings_matrix = np.load('emb.npy')

    featuresmine = np.load('../data/profiles_features.npy')
    featuresmine = np.log10(featuresmine+1e-4)

    Bert_rep = np.load('../data/Bert_encode.npy')
    features_itfidf = np.load('../data/features_IG_itfidf_gender.npy')
    print(features_itfidf.shape)
    features = np.concatenate([features_itfidf, featuresmine], axis= 1)
    #In[]

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
        flatt = None
        if model == "RNNS-RNNW-Feat-B":
            flatt = K.layers.concatenate([rnncell_seq, rnncell_sentences, f, B], name='concatenate')
        elif model == "RNNS-RNNW-B":
            flatt = K.layers.concatenate([rnncell_seq, rnncell_sentences, B], name='concatenate')
        elif model == "RNNS-Feat-B":
            flatt = K.layers.concatenate([rnncell_sentences, f, B], name='concatenate')
        elif model == "RNNW-Feat-B":
            flatt = K.layers.concatenate([rnncell_seq, f, B], name='concatenate')
        elif model == "B":
            flatt = B#K.layers.concatenate([rnncell_seq, rnncell_sentences, f, B], name='concatenate')
        
        # flatt = K.layers.concatenate([rnncell_seq, rnncell_sentences, f, B])
        flatt = K.layers.Dense(units=64, name='dense', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.04))(flatt)
        flatt = K.layers.LeakyReLU(alpha=0.3)(flatt)
        
        Y = K.layers.Dense(2, activation='softmax', name='densex')(flatt)
        return K.Model(inputs=[I, S, F, B], outputs=Y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 7)

    acc = 0

    f1mean = 0
    for i, (train_index, test_index) in enumerate(skf.split(encode_seqs, labels)):                    
        K.backend.clear_session()
        np.random.seed(13)
        tf.random.set_seed(13)
        checkpointer = K.callbacks.ModelCheckpoint('sex_model_[B|itfidf_Mine].h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True,
                                            save_weights_only=True)

        Classifier = Model(encode_seqs[0].shape, sentences_encode[0].shape, features[0].shape, Bert_rep[0].shape)
        Classifier.compile(optimizer=K.optimizers.Adam(lr=2e-3),loss=K.losses.CategoricalCrossentropy(), metrics=['acc', f1])

        Y = K.utils.to_categorical(labels[train_index])
        Y_Val = K.utils.to_categorical(labels[test_index])
        history = Classifier.fit((encode_seqs[train_index,:], sentences_encode[train_index,:], features[train_index, :], Bert_rep[train_index, :]), Y,
                    validation_data=((encode_seqs[test_index, :], sentences_encode[test_index,:], features[test_index,:], Bert_rep[test_index,:]), Y_Val)
                    , epochs=16, batch_size=64, callbacks=[checkpointer])
        
        z = np.argmax(history.history['val_acc'])
        f1mean += history.history['val_f1'][z]        
        break
    # print(f1mean/5)
    # with open('gender_f1', 'a') as file:
    #     file.write(model + ": " + str(f1mean/5) + "\n")
            

    # print(acc/5)
    # # %%
    # Classifier.load_weights('sex_model_[B|itfidf_Mine].h5', by_name=True)
    # y = Classifier.predict((encode_seqs[test_index, :], sentences_encode[test_index,:], features[test_index, :], Bert_rep[test_index, :])).argmax(axis=1)
    # print(y.shape, labels[test_index].shape)
    # print((y == labels[test_index]).sum()/len(y))
    # matrix = sklearn.metrics.confusion_matrix(labels[test_index], y)
    # ax= plt.subplot()
    # seaborn.heatmap(matrix, cmap="YlGnBu",annot=True, ax = ax, square=False,robust=True, fmt='.4g')

    # # # labels, title and ticks
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    # plt.show()

