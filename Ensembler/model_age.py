# %%
import sys
sys.path.append('../')
from toolsIT import *
import tensorflow as tf
import keras as K
import numpy as np
import keras_self_attention
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import plot_confusion_matrix, f1_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics
import seaborn
from keras import backend as bk
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Precompute Weights for Tweet Representation')
    parser.add_argument('-md', metavar='-model', required=True, help='Model Composition')
    parser.add_argument('-dp', metavar='-datapath', required=True, help='Trainning Data Path')
    args = parser.parse_args()
    model = args.md
        
    EMBD_PATH = "../data/wiki-it.vec"
    DATA_PATH = args.dp 

    embeddings_matrix, dicc = read_embedding(EMBD_PATH)
    encode_seqs, labels, _ = load_for_age(embeddings_matrix, dicc, DATA_PATH, 'age', 'train')

    # np.save('labls_age.npy', labels)
    # np.save('seqs_age.npy', encode_seqs)

    # x = np.save('emb', embeddings_matrix)
    encode_sentences = np.load('../data/profiles_sentence_encode.npy')
    # labels = np.load('labls_age.npy')
    # encode_seqs = np.load('seqs_age.npy')
    # embeddings_matrix = np.load('emb.npy')
    print('Examples Shape:', encode_seqs.shape)
    print('Labels Shape: ', labels.shape)

    # %%
    Bert_feat = np.load('../data/Bert_encode.npy')
    featuresmine = np.load('../data/profiles_features.npy')
    featuresmine = np.log10(featuresmine+1e-4)
        
    features_itfidf = np.load('../data/features_IG_itfidf_age.npy')
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
        
        flatt = K.layers.Dense(units=64, activation=None, name='preout')(flatt)
        flatt = K.layers.LeakyReLU(alpha=0.2)(flatt)

        Y = K.layers.Dense(5, activation='softmax', name='densex')(flatt)
        return K.Model(inputs=[I, S, F, B], outputs=Y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 7)


    
    f1mean = 0
    for i, (train_index, test_index) in enumerate(skf.split(encode_seqs, labels)):    
        K.backend.clear_session()
        np.random.seed(29)
        tf.random.set_seed(29)
        checkpointer = K.callbacks.ModelCheckpoint('ages_model_[B|itfidf_Mine].h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True,
                                            save_weights_only=True)

        Classifier = Model(encode_seqs[0].shape, encode_sentences[0].shape, features[0].shape, Bert_feat[0].shape)
        Classifier.compile(optimizer=K.optimizers.Adam(lr=2e-3, decay = 5e-5), 
                        loss=K.losses.CategoricalCrossentropy(), metrics=['acc', f1])

        # print(Classifier.summary())
        # exit()
        Y = K.utils.to_categorical(labels[train_index])
        Y_Val = K.utils.to_categorical(labels[test_index])
        history = Classifier.fit((encode_seqs[train_index,:], encode_sentences[train_index,:], features[train_index, :], Bert_feat[train_index, :]), Y, 
            validation_data=((encode_seqs[test_index, :], encode_sentences[test_index, :], features[test_index,:], Bert_feat[test_index, :]), Y_Val), 
                    epochs=16, batch_size=64, callbacks=[checkpointer])
        

        z = np.argmax(history.history['val_acc'])
        f1mean += history.history['val_f1'][z]        
        break
    # print(f1mean/5)
    # with open('age_f1', 'a') as file:
    #     file.write(model + ": " + str(f1mean/5) + "\n")

    # # %%    Representation 
    # Classifier.load_weights('ages_model_[B|itfidf_Mine].h5', by_name=True)
    # layer_output = Classifier.get_layer('densex').output
    # New_Embbeder = K.models.Model(inputs=Classifier.input, outputs=layer_output)
    # M = New_Embbeder.predict((encode_seqs[test_index, :], encode_sentences[test_index, :], features[test_index, :], Bert_feat[test_index, :]))
    # # M = np.random.rand(944, 1683)
    # print(M.shape)
    # pca = PCA(n_components=3)
    # M = pca.fit_transform(M)

    # dic = {}
    # print(M.shape)
    # for i in range(5):
    #     dic[i] = []
    # for i in range(len(M)):
    #     dic[labels[i]].append(M[i])

    # fig = plt.figure()

    # ax = fig.add_subplot(111, projection='3d')
    # colors = ['b', 'g', 'r', 'y', 'w']

    # for i in range(5):
    #     dic[i] = np.array(dic[i])
    #     dic[i] = dic[i].transpose()
    #     ax.scatter(dic[i][0], dic[i][1], dic[i][2],c = colors[i], label = ['0-19', '20-29', '30-39', '40-49', '50-100'][i])

    # plt.show()
    # # %%
    # y = Classifier.predict([encode_seqs[test_index,:], encode_sentences[test_index, :], features[test_index,:], Bert_feat[test_index,:]]).argmax(axis=1)
    # print((y == labels[test_index]).sum()/len(y))
    # matrix = sklearn.metrics.confusion_matrix(labels[test_index], y)
    # ax= plt.subplot()
    # seaborn.heatmap(matrix, cmap="YlGnBu",annot=True, ax = ax, square=False,robust=True, fmt='.4g')
    
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    # plt.show()
