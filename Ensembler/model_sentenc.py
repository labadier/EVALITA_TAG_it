import sys
import os
sys.path.append('../')

from toolsIT import *
from matplotlib import pyplot as plt
import tensorflow as tf
import keras as K
from sklearn.model_selection import StratifiedKFold
import time

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Precompute Weights for Tweet Representation')

    parser.add_argument('-dp', metavar='-data_path', required=True, help='The Path of the data')
    args = parser.parse_args()

    EMBD_PATH = "../data/wiki-it.vec"
    DATA_PATH = args.dp

    to_encode, seqs, ages, topic, sex, embeddings_matrix, _ = get_data_to_my_encoder(DATA_PATH, EMBD_PATH, 'train')
    x = np.concatenate((ages, sex, topic), axis=1)
    examples = np.sum(seqs)
    train_encoder = np.zeros((examples, 100))
    labels_encoder = np.zeros((examples, x.shape[1]))

    ide = 0
    idep = 0
    for i in to_encode:
        for j in range(seqs[idep]):
            train_encoder[ide] += i[j]
            labels_encoder[ide] = x[idep]
            ide += 1
        idep += 1
        
    np.save('data', train_encoder)
    np.save('labels', labels_encoder)
    # train_encoder = np.load('data')
    # labels_encoder = np.load('labels')

    embd_layer = K.layers.Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1],
        embeddings_initializer=K.initializers.Constant(embeddings_matrix), input_length=100, trainable=False, name='embd')
        
    def Model(inputshape):
        X = K.layers.Input(inputshape, name='input')
        E = embd_layer(X)
        print('seq: ', E.shape)
        X1 = K.layers.Conv1D(32, 5, padding='valid', data_format='channels_last', name='conv1M')(E)
        X1 = K.layers.Dropout(rate=0.2, name='dp0M')(X1)
        X1 = K.layers.BatchNormalization(axis=2, name='normlay1M')(X1)
        X1 = K.layers.Activation(activation='relu', name='Relu1M')(X1)
        X1 = K.layers.MaxPooling1D(pool_size=5, name='maxpooling1M', padding='valid')(X1)
        print('X1.shape', X1.shape)

        X2 = K.layers.Conv1D(32, 4, name='conv2M', padding='valid')(E)
        X2 = K.layers.Dropout(rate=0.2, name='dp2M')(X2)
        X2 = K.layers.BatchNormalization(axis=2, name='normlay2M')(X2)
        X2 = K.layers.Activation(activation='relu', name='Relu2M')(X2)
        X2 = K.layers.MaxPooling1D(4, name='maxpooling2M', padding='valid')(X2)
        print('X2.shape', X2.shape)

        X3 = K.layers.Conv1D(32, 3, name='conv3M', padding='valid')(E)
        X3 = K.layers.Dropout(rate=0.2, name='dp3M')(X3)
        X3 = K.layers.BatchNormalization(axis=2, name='normlay3M')(X3)
        X3 = K.layers.Activation(activation='relu', name='Relu3M')(X3)
        X3 = K.layers.MaxPooling1D(3, name='maxpooling3M', padding='valid')(X3)
        print('X3.shape', X3.shape)

        X4 = K.layers.concatenate([X1, X2, X3], name='concatenateconvwordsM', axis = 1)
        print('X4.shape', X4.shape)

        X4 = K.layers.LSTM(64, return_sequences=False, name='dblstm')(X4)
        X4 = K.layers.Dropout(0.2, name='dprm1')(X4)

        X4 = K.layers.Dense(32, activation=None, name='rmd1')(X4)
        X4 = K.layers.LeakyReLU(alpha=0.1, name='activation_dense')(X4)
        X4 = K.layers.BatchNormalization(name='bn1stdense')(X4)

        A = K.layers.Dense(5, activation='softmax', name='hyatage')(X4)
        S = K.layers.Dense(2, activation='softmax', name='hyat2sex')(X4)
        T = K.layers.Dense(11, activation='softmax', name='hyat2topic')(X4)
        return K.Model(inputs=X, outputs=[A, S, T])


    epoches = 0
    lrate = 0.002

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 7)

    acc = 0
    for i, (train_index, test_index) in enumerate(skf.split(train_encoder, labels_encoder[:,1])):    
        K.backend.clear_session()
        np.random.seed(1)
        M = Model(train_encoder[0].shape)
        M.compile(optimizer=K.optimizers.Adam(lr=lrate, decay=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        filepath = '../data/tp_EncoderConv_LSTM.h5'
        checkpointer = K.callbacks.ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True,
                                                save_weights_only =True, verbose=True)

        label = labels_encoder
        a = K.utils.to_categorical(labels_encoder[:,0])
        s = K.utils.to_categorical(labels_encoder[:,1])
        t = K.utils.to_categorical(labels_encoder[:,2])
        M.fit(train_encoder[train_index], [a[train_index], s[train_index], t[train_index]], 
            validation_data=(train_encoder[test_index], [a[test_index], s[test_index], t[test_index]]),
            epochs=20, batch_size=64, callbacks=[ checkpointer])
        break