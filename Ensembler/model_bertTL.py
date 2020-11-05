import sys
sys.path.append('../')
import keras_bert
import tensorflow as tf
import keras as K
from bert.tokenization import FullTokenizer
from toolsIT_bert import *
import time
from matplotlib import pyplot as plt
import os


#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Precompute Weights for Tweet Representation')

    parser.add_argument('-dp', metavar='-data_path', required=True, help='The Path of the data')
    args = parser.parse_args()

    DATA_PATH = args.dp

    def convert_lines(to_encode, max_seq_length, tokenizer):
        ide = 0
        truncated = 0
        test = np.zeros((len(to_encode), max_seq_length))
        for i in range(to_encode.shape[0]):
            sentence = tokenizer.tokenize(' '.join(to_encode[i]))
            if len(sentence) > max_seq_length - 2:
                sentence = sentence[:max_seq_length - 2]
                truncated += 1
            test[i] = tokenizer.convert_tokens_to_ids(["[CLS]"] + sentence + ["[SEP]"]) + [0] * (
                    max_seq_length - 2 - len(sentence))

        print('Exampes Truncated: ', truncated)
        return test

    maxlen = 350
    BERT_PRETRAINED_DIR = "../data/multi_cased_L-12_H-768_A-12/"
    os.path.dirname(os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json'))

    config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    model = keras_bert.loader.load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, seq_len=maxlen)

    np.random.seed(17)
    tf.random.set_seed(17)
    sequence_output = model.get_layer('Encoder-2-FeedForward-Norm').output
    extractF = keras_bert.layers.Extract(0, name='Extract_First')(sequence_output)
    extractL = keras_bert.layers.Extract(99, name='Extract_Last')(sequence_output)

    print('Extracted Vector: ', extractL.shape)
    print('Extracted Vector: ', extractF.shape)
    extract = K.layers.concatenate([extractF, extractL], axis = 1, name='Conscatenate_BERT_out')
    print('Extracted Vector: ', extract.shape)

    pool_output = K.layers.Dense(64, activation='relu', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                                , name='encoder_layer')(extract)

    pool_output_sex = K.layers.Dense(2, activation='softmax', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                                , name='real_output_sex')(pool_output)
    pool_output_age = K.layers.Dense(5, activation='softmax', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                                , name='real_output_age')(pool_output)
    pool_output_topic = K.layers.Dense(11, activation='softmax', kernel_initializer=K.initializers.TruncatedNormal(stddev=0.02)
                                , name='real_output_topic')(pool_output)

    mymodel = K.models.Model(inputs=model.input, outputs=[pool_output_age, pool_output_topic, pool_output_sex])
    mymodel.compile(loss=K.losses.categorical_crossentropy, optimizer=K.optimizers.Adam(lr=1e-5, decay=2e-5),
                    metrics=["acc"])
    # mymodel.summary(line_length=170)
 
    to_encode, labels = load_for_age(DATA_PATH, 'train')
    dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
    tokenizer = FullTokenizer(vocab_file=dict_path, do_lower_case=True)
    test = convert_lines(to_encode, maxlen, tokenizer)
    # np.save('./data/tmp_sequences_unlemad.npy', test)

    # test = np.load('./data/tmp_sequences_unlemad.npy')
    # labels = np.load('./data/tmp_original_labls.npy')[:,16]
    # labels = K.utils.to_categorical(labels)

    print('Exampes', test.shape[0])
    seg_input = np.zeros((test.shape[0], maxlen))
    mask_input = np.ones((test.shape[0], maxlen))
    
    perm = np.random.permutation(test.shape[0])
    test = test[perm, :]
    labels = labels[perm, :]
    
    filepath = '../data/BERT.h5'
    checkpointer = K.callbacks.ModelCheckpoint(filepath, monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only =True)
    mymodel.fit([test, seg_input, mask_input],[K.utils.to_categorical(labels[:,0]), K.utils.to_categorical(labels[:,1]), K.utils.to_categorical(labels[:,2])], batch_size=16, epochs=2, validation_split=0.1, callbacks=[checkpointer])
