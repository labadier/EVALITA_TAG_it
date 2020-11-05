from toolsIT import *
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf
import sklearn.svm as svm
from sklearn.model_selection import StratifiedKFold
from toolsIT import load_data as ld
from toolsIT import Preprocess_IG
from toolsIT import Preprocess as pp
import os

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Precompute Weights for Tweet Representation')

    parser.add_argument('-d', metavar='-data_path', required=True, help='The Path of the data')
    args = parser.parse_args()
    DATA_PATH = args.d

    stop_words = []

    # task = 'topic'
    len_class = {'age':AGE_CLASSES, 'topic':TOPIC_CLASSES, 'gender':SEX_CLASSES}
    # tope = int(500/len(len_class[task]))

    with open('./data/stopwords-it.txt', mode='rU') as itsw:    
        for wordors in itsw:
            stop_words.append(wordors[:-1])

    profiles, postxuser = ld(DATA_PATH) 
    profiles = pp(profiles, 'lemma')

    def get_vocab(tope):
        documents = []
        vocabulary = []
        for i in AGE_CLASSES:
            documents.append([])
            documents[-1] = [ x['post'] for x in profiles if x['age'] == i]
            Bow = TfIdf(stop_words=stop_words, max_features=tope, sublinear_tf=True)
            _ = Bow.fit_transform(documents[-1])
            vocabulary += Bow.get_feature_names()
        np.save('./data/vocab', np.array(vocabulary))


    def Compute_IG(task):
        TC = [ dict() for i in len_class[task] ] # term frequency by class
        FC = [ 0 for i in len_class[task] ] #class frequency
        CC = [ 0 for i in len_class[task] ] #class cardinality
        tt = 0 #terms quantity
        GTF = dict() #global term frequency
        tc = len(profiles) #classes quantity
        IG = [ [] for i in len_class[task] ] #****** Information Gain

        for j in profiles:

            FC[len_class[task].index(j[task])] += 1
            for x in j['post']:
                if stop_words.count(x) > 0:
                    continue
                if TC[len_class[task].index(j[task])].get(x) is None:
                    TC[len_class[task].index(j[task])][x] = 1
                TC[len_class[task].index(j[task])][x] += 1
                CC[len_class[task].index(j[task])] += 1 
                tt += 1
                if GTF.get(x) is None:
                    GTF[x] = 1
                GTF[x] += 1

        for c in range(len(len_class[task])):
            for t in TC[c].keys():

                TkC = TC[c][t]/CC[c] 
                TkC = TkC * np.log2( 1e-5 + TkC/( (GTF[t]/tt)*(FC[c]/tc) ) )

                TkCh = (GTF[t] - TC[c][t])/(tt - CC[c])
                TkCh = TkCh * np.log2( 1e-5 + TkCh/( (GTF[t]/tt)*(1.0 - FC[c]/tc) ))

                TkhC = 1.0 - TC[c][t]/CC[c] 
                TkhC = TkhC * np.log2( 1e-5 + TkhC/( (1.0 - GTF[t]/tt)*(FC[c]/tc) ) )

                TkhCh = (tt - CC[c] - GTF[t] + TC[c][t])/(tt - CC[c])
                TkhCh = TkhCh * np.log2( 1e-5 + TkhCh/((1.0 - GTF[t]/tt)*(1.0 - FC[c]/tc)))
                IG[c].append((TkC + TkCh + TkhC + TkhCh, t))
                IG[c].sort(reverse = True)
        return IG

    # IG = Compute_IG(task)
    # vocabulary = []
    # for c in range(len(len_class[task])):
    #     for i in range(tope):
    #         vocabulary.append(IG[c][i][1])
    # np.save('vocab_' + task + '.npy', vocabulary)

    #%%
    # get_vocab(tope)
    for task in ['gender', 'topic', 'age']:
        tope = int(500/len(len_class[task]))
        vocabulary = np.load('./data/vocab_'+task+'.npy')#'./weights/age_model/vocab.npy')
        features = []
        documents =  [ x['post'] for x in profiles]
        for i in range(len(len_class[task])):
            BoW = TfIdf(stop_words=stop_words, vocabulary=list(vocabulary[i*tope:(i + 1)*tope]), sublinear_tf=True)
            X = BoW.fit_transform(documents)
            features.append(X.toarray())
        features = np.concatenate(features, axis=1)
        np.save('./data/features_IG_itfidf_'+task, features)
    os.system('clear')
    print('Features ITFIDF Extracted !!!')

    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 7)
    # acc = 0
    # for i, (train_index, test_index) in enumerate(skf.split(features, Y)):
    #     model = svm.LinearSVC()
    #     model.fit(features[train_index,:], Y[train_index])
    #     y_pred = model.predict(features[test_index, :])
    #     y = Y[test_index]
    #     acc += (y_pred == y).sum() / y_pred.shape[0]
    #     print((y_pred == y).sum() / y_pred.shape[0])
    # print(acc/skf.n_splits)
    # np.save('/home/nitro/projects/EVALITA/test_data/features_IG_itfidf_'+task, features)
