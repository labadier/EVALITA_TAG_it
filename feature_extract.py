#%%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf
import sklearn.svm as svm
from sklearn.model_selection import StratifiedKFold
from toolsIT import *
import os

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Precompute Weights for Tweet Representation')

    parser.add_argument('-d', metavar='-data_path', required=True, help='The Path of the data')
    args = parser.parse_args()
    DATA_PATH = args.d
        
    def LD(data_path):
        postxuser = []
        print(data_path)
        XTree = open(data_path, mode='rU')
        users = []
        band = False
        for line in XTree:
            usr = line.split(' ')

            if usr[0] == "<user":
                users.append({'post': [], 'id': int(usr[1][4:len(usr[1]) - 1]), 'age': usr[3][len("topic"):len(usr[3]) - 1],
                            'gender': usr[4][len("gender= "):len(usr[4]) - 3],
                            'topic': usr[2][len("topic= "):len(usr[2]) - 1]})
                postxuser.append(0)
            if band:
    #             users[len(users) - 1]['post'] += " < post > "
                if line[:len(line) - 1] == '':
                    users[len(users) - 1]['post'].append("<vuoto> .")
                else:
                    tmp = line[:len(line) - 1]
                    for k in emoji.keys():
                        while True:
                            z = str.find(tmp, k)
                            if z == -1:
                                break
                            tmp1 = tmp[z + len(k):]
                            tmp = tmp[:z] + '<emoticon> ' + emoji[k] + ' '
                            tmp += tmp1
                    users[len(users) - 1]['post'].append(str(tmp))
                if users[len(users) - 1]['post'][len(users[len(users) - 1]['post']) - 1][-1] not in {'?', '!', '.', '...'}:
                    users[len(users) - 1]['post'][-1] += "."
                band = False

            if usr[0][:len(usr[0]) - 1] == "<post>":
                band = True
                postxuser[len(postxuser) - 1] += 1
        return users, postxuser

    def FP(users):
        
        print('Freeling Tokenizing ', '0 %\r', end="")

        if "FREELINGDIR" not in os.environ:
            os.environ["FREELINGDIR"] = "/usr/local"
        DATA = os.environ["FREELINGDIR"] + "/share/freeling/";

        pyfreeling.util_init_locale("default");
        LANG = "es";
        op = pyfreeling.maco_options(LANG);
        op.set_data_files("",
                        DATA + "common/punct.dat",
                        DATA + LANG + "/dicc.src",
                        DATA + LANG + "/afixos.dat",
                        "",
                        DATA + LANG + "/locucions.dat",
                        DATA + LANG + "/np.dat",
                        "",  # DATA + LANG + "/quantities.dat",
                        DATA + LANG + "/probabilitats.dat");

        tk = pyfreeling.tokenizer(DATA + LANG + "/tokenizer.dat");
        sp = pyfreeling.splitter(DATA + LANG + "/splitter.dat");
        mf = pyfreeling.maco(op);
        mf.set_active_options(False, True, True, True,  # select which among created
                            True, True, True, True,  # submodules are to be used.
                            True, True, False, True);  # default: all created submodules are used

        tg = pyfreeling.hmm_tagger(DATA + LANG + "/tagger.dat", True, 2);
        sen = pyfreeling.senses(DATA + LANG + "/senses.dat");

        done = 0
        perc = 0
        top = len(users)
        cont = 0;
        senten = []
        adj_sust_punt = np.zeros((len(users), 3))
        for i in range(len(users)):
            senten.append([])
            done += 1
            z = done / top
            z *= 100
            z = int(z)
            if z - perc >= 1:
                perc = z
                print('Freeling Tokenizing ', str(perc) + ' %\r', end="")

            for j in range(len(users[i]['post'])):
                senten[-1].append(0)
                x = users[i]['post'][j]
                l = tk.tokenize(x);
                #         sid=sp.open_session();
                ls = sp.split(l);

                ls = mf.analyze(ls);
                ls = tg.analyze(ls);
                ls = sen.analyze(ls);

                listsent = []
                for s in ls:
                    senten[-1][-1] += 1
                    cont  += 1
                    ora = s.get_words()
                    for k in range(len(ora)):
                        if ora[k].get_tag() == 'W':
                            listsent.append( 'data' )
                        elif ora[k].get_tag() == 'Z':
                            listsent.append( 'numero' )
                        else:
                            listsent.append(ora[k].get_form().lower())
    #                     print(ora[k].get_form().lower(), ora[k].get_tag())
    #                 listsent.append(tmp)
                        if ora[k].get_tag()[0] == 'N':
                            adj_sust_punt[i][0] += 1
                        if ora[k].get_tag()[0] == 'A':
                            adj_sust_punt[i][1] += 1
                        if ora[k].get_tag()[0] == 'F':
                            adj_sust_punt[i][2] += 1
                users[i]['post'][j] = np.array(listsent)

        print('Freeling Tokenizing ok        ')
        return users, adj_sust_punt, senten

    profiles, postxuser = LD(DATA_PATH) 
    features = np.zeros((len(profiles), 12))

    profiles, adj_sust_punt, senten = FP(profiles)
    # tmp = [{'post':['la vida! es bella...', 'el sol está caliente.']}]
    # _, adj_sust_punt = Preprocess(tmp)
    stop_words = []
    itsw =  open('./data/stopwords-it.txt', mode='rU')
    for wordors in itsw:
        stop_words.append(wordors[:-1])
        
    for i in range(len(profiles)):
        
        emojis = 0
        paldif = set()
        ptos = 0
        sentdev = np.zeros((len(profiles[i]['post']), 1))
        longdev = np.zeros((len(profiles[i]['post']), 1))
        for j in range(len(profiles[i]['post'])):
            emojis += len(np.where(profiles[i]['post'][j]=='emoticon')[0])
            features[i][8] += len(np.where(profiles[i]['post'][j]=='vuoto')[0]) > 0
            sentdev[j] = senten[i][j]
            longdev[j] = len(profiles[i]['post'][j])
            
            for k in profiles[i]['post'][j]:
                paldif.add(k)

        for j in paldif:
            if(stop_words.count(j)):
                features[i][9] += 1
            
        features[i][0] = len(profiles[i]['post'])
        features[i][10] = adj_sust_punt[i][2]
        features[i][4] = adj_sust_punt[i][0]
        features[i][3] = adj_sust_punt[i][1]
        features[i][5] = emojis
        features[i][11] = len(paldif)
        
        features[i][6] = np.mean(sentdev)
        features[i][7] = np.std(sentdev) 

        features[i][2] = np.mean(longdev)
        features[i][1] = np.std(longdev)
    np.save('./data/profiles_features', features)
    os.system('clear')
    print('Stylistic Features Extracted !!!')