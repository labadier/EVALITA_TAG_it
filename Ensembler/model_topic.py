#%%
import sys
sys.path.append('../')
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


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Precompute Weights for Tweet Representation')
    parser.add_argument('-md', metavar='-model', required=True, help='Model Composition')
    parser.add_argument('-dp', metavar='-datapath', required=True, help='Trainning Data Path')
    args = parser.parse_args()
    model = args.md
        
    EMBD_PATH = "../data/wiki-it.vec"
    DATA_PATH = args.dp

    stop_words = []
    itsw =  open('../data/stopwords-it.txt', mode='rU')
    for wordors in itsw:
        stop_words.append(wordors[:-1])

    embeddings_matrix, dicc = read_embedding(EMBD_PATH)
    encode_seqs, labels, _ = lematized_for_topic(embeddings_matrix, dicc, DATA_PATH, 'train', stop_words)

    # np.save('labls_topic.npy', labels)
    # np.save('seqs_topic.npy', encode_seqs)

    snetence_encodes = np.load('../data/profiles_sentence_encode.npy')
    # labels = np.load('labls_topic.npy')
    # encode_seqs = np.load('seqs_topic.npy')
    # embeddings_matrix = np.load('emb.npy')

    print('Examples Shape:', encode_seqs.shape)
    print('Labels Shape: ', labels.shape)
    #%%

    featuresmine = np.load('../data/profiles_features.npy')
    Bert_feat = np.load('../data/Bert_encode.npy')

    featuresmine = np.log10(featuresmine+1e-4)
        
    features_itfidf = np.load('../data/features_IG_itfidf_topic.npy', allow_pickle=True)
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
        
        flatt = None
        if model == "RNNS-RNNW-Feat-B":
            flatt = K.layers.concatenate([rnncell_seq, rnncell_seneteces, f, B], name='concatenate')
        elif model == "RNNS-RNNW-B":
            flatt = K.layers.concatenate([rnncell_seq, rnncell_seneteces, B], name='concatenate')
        elif model == "RNNS-Feat-B":
            flatt = K.layers.concatenate([rnncell_seneteces, f, B], name='concatenate')
        elif model == "RNNW-Feat-B":
            flatt = K.layers.concatenate([rnncell_seq, f, B], name='concatenate')
        elif model == "B":
            flatt = B#K.layers.concatenate([rnncell_seq, rnncell_sentences, f, B], name='concatenate')
        
        # flatt = K.layers.concatenate([rnncell_seq, rnncell_seneteces, f, B])
        flatt = K.layers.Dense(units=64, name='dense')(flatt)
        flatt = K.layers.LeakyReLU(alpha=0.1)(flatt)
        
        Y = K.layers.Dense(11, activation='softmax', name='densex_')(flatt)
        
        return K.Model(inputs=[I, S, F, B], outputs=Y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 11)#7)

    acc = 0
    iter = 0
    for i, (train_index, test_index) in enumerate(skf.split(encode_seqs, labels)):  
        
        iter += 1
        K.backend.clear_session()
        np.random.seed(7)
        tf.random.set_seed(7)
        checkpointer = K.callbacks.ModelCheckpoint('topic_model_[B|itfidf_Mine].h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True,
                                            save_weights_only=True)
        Classifier = Model(encode_seqs[0].shape, snetence_encodes[0].shape,features[0].shape, Bert_feat[0].shape)
        Classifier.compile(optimizer=K.optimizers.Adam(lr=3e-3 , decay = 1e-5), 
                        loss=K.losses.CategoricalCrossentropy(), metrics=['acc'])
        #print(Classifier.summary())
        #exit()
        Y = K.utils.to_categorical(labels[train_index])
        Y_Val = K.utils.to_categorical(labels[test_index])
        history = Classifier.fit((encode_seqs[train_index,:], snetence_encodes[train_index,:], features[train_index, :], Bert_feat[train_index, :]), Y, 
            validation_data=((encode_seqs[test_index, :], snetence_encodes[test_index,:], features[test_index,:], Bert_feat[test_index,:]), Y_Val), 
                    epochs=16, batch_size=64, callbacks=[checkpointer])
        z = np.max(history.history['val_acc'])
        acc += z
        break

    # print(acc/5)
    # with open('topic_acc', 'a') as file:
    #     file.write(model + ": " + str(acc/5) + "\n")

    # # %%
    # # Classifier.load_weights('topic_model_[B|itfidf_Mine].h5', by_name=True)
    # Y = Classifier.predict((encode_seqs[test_index, :], snetence_encodes[test_index,:], features[test_index,:], Bert_feat[test_index,:]))
    # Y = np.argmax(Y, axis=1)
    # acc = (Y == labels[test_index]).sum()/len(test_index)
    # print(acc)
    # matrix = sklearn.metrics.confusion_matrix(labels[test_index], Y)
    # ax= plt.subplot()
    # seaborn.heatmap(matrix, cmap="YlGnBu", annot=True, ax = ax, square=False,robust=True, fmt='.4g')
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    # plt.show()
