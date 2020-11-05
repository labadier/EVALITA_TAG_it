import numpy as np
import pyfreeling
import os
import json
from keras import backend as bk

emoji = {':-)': 'sorridi', '(-:': 'sorridi', ':)': 'sorridi', '(:': 'sorridi', '=]': 'sorridi', ':]': 'sorridi',
         ':d': 'ridere', '8d': 'ridere', 'xd': 'ridere', '=d': 'ridere', ':-d': 'ridere',
         ':D': 'ridere', '8D': 'ridere', 'xD': 'ridere', '=D': 'ridere', ':-D': 'ridere',
         'x-d': 'ridere', ':(': 'triste', '):': 'triste', ':c': 'triste', ':[': 'triste', ':\'(': 'pianto', ':,( ': 'pianto',
         'x-D': 'ridere', ':(': 'triste', '):': 'triste', ':C': 'triste', ':[': 'triste', ':\'(': 'pianto', ':,( ': 'pianto',
         ':"(': 'pianto', ':((': 'pianto', ':|': 'neutrale', '=_=': 'neutrale', '-_-': 'neutrale', ':-\\': 'neutrale',
         ':O': 'neutrale', ':-!': 'neutrale'}

AGE_CLASSES = ['0-19', '20-29', '30-39', '40-49', '50-100']
TOPIC_CLASSES = ['ANIME', 'AUTO-MOTO', 'BIKES', 'CELEBRITIES', 'ENTERTAINMENT', 'MEDICINE-AESTHETICS',
                 'METAL-DETECTING', 'NATURE', 'SMOKE', 'SPORTS', 'TECHNOLOGY']
SEX_CLASSES = ['M', 'F']
    
def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = bk.sum(bk.round(bk.clip(y_true * y_pred, 0, 1)))
            possible_positives = bk.sum(bk.round(bk.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + bk.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = bk.sum(bk.round(bk.clip(y_true * y_pred, 0, 1)))
            predicted_positives = bk.sum(bk.round(bk.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + bk.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+bk.epsilon()))

def load_stylistic_features(STYLE_PATH):
    with open(STYLE_PATH) as F:
        data = json.load(F) 
    STYF = [ np.fromstring(data['training']['txt'][str(i+1)]['vec'], 'f', sep=',') for i in range(len(data['training']['txt']))]
    return np.array(STYF)

def load_data(data_path):
    postxuser = []
    print(data_path)
    XTree = open(data_path, mode='rU')
    users = []
    band = False
    for line in XTree:
        usr = line.split(' ')

        if usr[0] == "<user":
            users.append({'post': '', 'id': usr[1][4:len(usr[1]) - 1], 'age': usr[3][len("topic"):len(usr[3]) - 1],
                          'gender': usr[4][len("gender= "):len(usr[4]) - 3],
                          'topic': usr[2][len("topic= "):len(usr[2]) - 1]})
            postxuser.append(0)
        if band:
            users[len(users) - 1]['post'] += " < post > "
            if line[:len(line) - 1] == '':
                users[len(users) - 1]['post'] += " < vuoto > ."
            else:
                tmp = line[:len(line) - 1]
                for k in emoji.keys():
                    while True:
                        z = str.find(tmp, k)
                        if z == -1:
                            break
                        tmp1 = tmp[z + len(k):]
                        tmp = tmp[:z] + '< emoticon > ' + emoji[k] + ' '
                        tmp += tmp1
                users[len(users) - 1]['post'] += tmp

            if users[len(users) - 1]['post'][len(users[len(users) - 1]['post']) - 1] not in {'?', '!', '.', '...'}:
                users[len(users) - 1]['post'] += '.'
            users[len(users) - 1]['post'] += ' '
            band = False

        if usr[0][:len(usr[0]) - 1] == "<post>":
            band = True
            postxuser[len(postxuser) - 1] += 1
    return users, postxuser

def Preprocess(users, type):
    print('Freeling Tokenizing ', '0 %\r', end="")
    if "FREELINGDIR" not in os.environ:
        os.environ["FREELINGDIR"] = "/usr/local"
    DATA = os.environ["FREELINGDIR"] + "/share/freeling/"

    pyfreeling.util_init_locale("default")
    LANG = "it"
    op = pyfreeling.maco_options(LANG)
    op.set_data_files("",
                      DATA + "common/punct.dat",
                      DATA + LANG + "/dicc.src",
                      DATA + LANG + "/afixos.dat",
                      "",
                      DATA + LANG + "/locucions.dat",
                      DATA + LANG + "/np.dat",
                      "",  # DATA + LANG + "/quantities.dat",
                      DATA + LANG + "/probabilitats.dat")

    tk = pyfreeling.tokenizer(DATA + LANG + "/tokenizer.dat")
    sp = pyfreeling.splitter(DATA + LANG + "/splitter.dat")
    mf = pyfreeling.maco(op)
    mf.set_active_options(False, True, True, True,  # select which among created
                          True, True, True, True,  # submodules are to be used.
                          True, True, False, True) # default: all created submodules are used

    tg = pyfreeling.hmm_tagger(DATA + LANG + "/tagger.dat", True, 2)
    sen = pyfreeling.senses(DATA + LANG + "/senses.dat")

    done = 0
    perc = 0
    top = len(users)
    cont = 0
    for i in range(len(users)):

        done += 1
        z = done / top
        z *= 100
        z = int(z)
        if z - perc >= 1:
            perc = z
            print('Freeling Tokenizing ', str(perc) + ' %\r', end="")

        x = users[i]['post']
        l = tk.tokenize(x)
        ls = sp.split(l)

        ls = mf.analyze(ls)
        ls = tg.analyze(ls)
        ls = sen.analyze(ls)
        
        listsent = []
        for s in ls:
            cont  += 1
            ora = s.get_words()
            tmp = []
            for k in range(len(ora)):

                if ora[k].get_tag() == 'W':
                    tmp.append( 'data' )
                elif ora[k].get_tag() == 'Z':
                    tmp.append( 'numero' )
                else:
                    if type == 'lemma':
                        tmp.append(ora[k].get_lemma().lower())
                    elif type == 'form':
                        tmp.append(ora[k].get_form().lower())
            listsent.append(' '.join(tmp))
        users[i]['post'] = ' '.join(listsent)

    print('Freeling Tokenizing ok        ')
    return users


def Preprocess_Sentence_Encoder(users, type):
    print('Freeling Tokenizing ', '0 %\r', end="")
    if "FREELINGDIR" not in os.environ:
        os.environ["FREELINGDIR"] = "/usr/local"
    DATA = os.environ["FREELINGDIR"] + "/share/freeling/"

    pyfreeling.util_init_locale("default")
    LANG = "it"
    op = pyfreeling.maco_options(LANG)
    op.set_data_files("",
                      DATA + "common/punct.dat",
                      DATA + LANG + "/dicc.src",
                      DATA + LANG + "/afixos.dat",
                      "",
                      DATA + LANG + "/locucions.dat",
                      DATA + LANG + "/np.dat",
                      "",  # DATA + LANG + "/quantities.dat",
                      DATA + LANG + "/probabilitats.dat")

    tk = pyfreeling.tokenizer(DATA + LANG + "/tokenizer.dat")
    sp = pyfreeling.splitter(DATA + LANG + "/splitter.dat")
    mf = pyfreeling.maco(op)
    mf.set_active_options(False, True, True, True,  # select which among created
                          True, True, True, True,  # submodules are to be used.
                          True, True, False, True) # default: all created submodules are used

    tg = pyfreeling.hmm_tagger(DATA + LANG + "/tagger.dat", True, 2)
    sen = pyfreeling.senses(DATA + LANG + "/senses.dat")

    done = 0
    perc = 0
    top = len(users)
    cont = 0
    for i in range(len(users)):

        done += 1
        z = done / top
        z *= 100
        z = int(z)
        if z - perc >= 1:
            perc = z
            print('Freeling Tokenizing ', str(perc) + ' %\r', end="")

        x = users[i]['post']
        l = tk.tokenize(x)
        ls = sp.split(l)

        ls = mf.analyze(ls)
        ls = tg.analyze(ls)
        ls = sen.analyze(ls)
        
        listsent = []
        for s in ls:
            cont  += 1
            ora = s.get_words()
            tmp = []
            for k in range(len(ora)):

                if ora[k].get_tag() == 'W':
                    tmp.append( 'data' )
                elif ora[k].get_tag() == 'Z':
                    tmp.append( 'numero' )
                else:
                    if type == 'lemma':
                        tmp.append(ora[k].get_lemma().lower())
                    elif type == 'form':
                        tmp.append(ora[k].get_form().lower())
            listsent.append(tmp)
        users[i]['post'] = np.array(listsent)

    print('Freeling Tokenizing ok        ')
    return users

def Preprocess_IG(users, type):
    print('Freeling Tokenizing ', '0 %\r', end="")
    if "FREELINGDIR" not in os.environ:
        os.environ["FREELINGDIR"] = "/usr/local"
    DATA = os.environ["FREELINGDIR"] + "/share/freeling/"

    pyfreeling.util_init_locale("default")
    LANG = "it"
    op = pyfreeling.maco_options(LANG)
    op.set_data_files("",
                      DATA + "common/punct.dat",
                      DATA + LANG + "/dicc.src",
                      DATA + LANG + "/afixos.dat",
                      "",
                      DATA + LANG + "/locucions.dat",
                      DATA + LANG + "/np.dat",
                      "",  # DATA + LANG + "/quantities.dat",
                      DATA + LANG + "/probabilitats.dat")

    tk = pyfreeling.tokenizer(DATA + LANG + "/tokenizer.dat")
    sp = pyfreeling.splitter(DATA + LANG + "/splitter.dat")
    mf = pyfreeling.maco(op)
    mf.set_active_options(False, True, True, True,  # select which among created
                          True, True, True, True,  # submodules are to be used.
                          True, True, False, True) # default: all created submodules are used

    tg = pyfreeling.hmm_tagger(DATA + LANG + "/tagger.dat", True, 2)
    sen = pyfreeling.senses(DATA + LANG + "/senses.dat")

    done = 0
    perc = 0
    top = len(users)
    cont = 0
    for i in range(len(users)):

        done += 1
        z = done / top
        z *= 100
        z = int(z)
        if z - perc >= 1:
            perc = z
            print('Freeling Tokenizing ', str(perc) + ' %\r', end="")

        x = users[i]['post']
        l = tk.tokenize(x)
        ls = sp.split(l)

        ls = mf.analyze(ls)
        ls = tg.analyze(ls)
        ls = sen.analyze(ls)
        
        listsent = []
        for s in ls:
            cont  += 1
            ora = s.get_words()
            for k in range(len(ora)):

                if ora[k].get_tag() == 'W':
                    listsent.append( 'data' )
                elif ora[k].get_tag() == 'Z':
                    listsent.append( 'numero' )
                else:
                    if type == 'lemma':
                        listsent.append(ora[k].get_lemma().lower())
                    elif type == 'form':
                        listsent.append(ora[k].get_form().lower())
        users[i]['post'] = np.array(listsent)

    print('Freeling Tokenizing ok        ')
    return users


def read_embedding(embed_path):
    word2v = open(embed_path, mode='rU')
    embeddings_index = {}
    dicc = {}
    o = 0
    embed_dim = -1

    band = True
    for line in word2v:
        word, coefs = line.split(maxsplit=1)
        if band:
            print(word, coefs)
            band = False
            continue
        coefs = np.fromstring(coefs, 'f', sep=' ')
        
        if coefs.shape != (300,):
            print(word)
            continue
            
        if dicc.get(word) is not None:
            embeddings_index[word] = (embeddings_index[word]+coefs)/2
            continue
            
        embeddings_index[word] = coefs
        dicc[word] = o
        o += 1
        embed_dim = max(embed_dim, embeddings_index[word].shape[0])

    embeddings_matrix = np.zeros((len(dicc) + 2, embed_dim))
    embeddings_matrix[len(dicc) + 1] = np.random.randn(1, embed_dim)
    #     print(embeddings_matrix.shape)

    for i in dicc.keys():
        embd = embeddings_index.get(i)
        embeddings_matrix[dicc[i]] += embd
    print('Read Embedding ok')
    return embeddings_matrix, dicc


def get_data(DATA_PATH, EMBD_PATH):
    profiles, postxuser = load_data(DATA_PATH) 
    profiles = Preprocess(profiles)

    TOP_SENT_LENGHT = 98
    SENTENCES = 30
    to_encode = np.array(np.zeros((len(profiles), SENTENCES, TOP_SENT_LENGHT)), dtype=np.str_)
    aproached_words = []
    losed_words = []
    rest_of_seq = []
    seqs = []
    ages = np.zeros((len(profiles), 5))
    sex = np.zeros((len(profiles), 1))  # 1-Female 0-Male
    topic = np.zeros((len(profiles), 11))
    ids = np.zeros((len(profiles), 1))

    pi = 0
    for i in profiles:
        
        ages[pi][AGE_CLASSES.index(profiles[pi]['age'])] = 1
        ids[pi] = profiles[pi]['id']
        topic[pi][TOPIC_CLASSES.index(profiles[pi]['topic'])] = 1
        if profiles[pi]['gender'] == 'F':
            sex[pi][0] = 1

        bag = []
        full = False
        rest_of_seq.append(0)
        aproached_words.append(0)
        losed_words.append(0)
        seqs.append(0)


        for j in i['post']:

            if full == True:
                rest_of_seq[len(rest_of_seq) - 1] += len(j)
                continue

            if len(bag) == 0:
                seqs[len(seqs)-1] += 1
                bag.append([])

            for k in j:
                if len(bag[len(bag) - 1]) >= TOP_SENT_LENGHT and len(bag) == SENTENCES:
                    rest_of_seq[len(rest_of_seq) - 1] += len(j)
                    break

                if len(bag[len(bag) - 1]) == TOP_SENT_LENGHT:
                    bag.append([])
                    seqs[len(seqs)-1] += 1
                bag[len(bag) - 1].append(k)

            if len(bag) == SENTENCES:
                break

        for j in range(len(bag)):
            for k in range(len(bag[j])):
                emb = bag[j][k]
                to_encode[pi][j][k] = bag[j][k]

            for k in range(len(bag[j]), TOP_SENT_LENGHT, 1):
                to_encode[pi][j][k] += '!*-/'
        pi += 1

    return  to_encode, seqs, ages, topic, sex, ids

def get_data_to_my_encoder(DATA_PATH, EMBD_PATH, mode):
    profiles, postxuser = load_data(DATA_PATH) 
    ids = np.array(np.zeros((len(profiles),)), dtype=np.str_)
    for i in range(len(profiles)):
        ids[i] = profiles[i]['id']
        
    if EMBD_PATH == '':
        return ids
    print('Data Loaded !!! ')
    profiles = Preprocess_Sentence_Encoder(profiles, 'form')

    embeddings_matrix, dicc = read_embedding(EMBD_PATH)
    PAD_TOKEN = len(dicc)
    UNK_TOKEN = len(dicc)+1
    
    TOP_SENT_LENGHT = 100
    SENTENCES = 30
    to_encode = np.zeros((len(profiles), SENTENCES, TOP_SENT_LENGHT))
    aproached_words = []
    losed_words = []
    rest_of_seq = []
    seqs = []
    ages = np.zeros((len(profiles), 1))
    sex = np.zeros((len(profiles), 1))  # 1-Female 0-Male
    topic = np.zeros((len(profiles), 1))

    pi = 0
    for i in profiles:
        if mode == 'train':
            ages[pi] = AGE_CLASSES.index(profiles[pi]['age'])
            topic[pi] = TOPIC_CLASSES.index(profiles[pi]['topic'])
            sex[pi] = SEX_CLASSES.index(profiles[pi]['gender'])

        bag = []
        full = False
        rest_of_seq.append(0)
        aproached_words.append(0)
        losed_words.append(0)
        seqs.append(0)


        for j in i['post']:

            if full == True:
                rest_of_seq[len(rest_of_seq) - 1] += len(j)
                continue

            if len(bag) == 0:
                seqs[len(seqs)-1] += 1
                bag.append([])

            for k in j:
                if len(bag[len(bag) - 1]) >= TOP_SENT_LENGHT and len(bag) == SENTENCES:
                    rest_of_seq[len(rest_of_seq) - 1] += len(j)
                    break

                if len(bag[len(bag) - 1]) == TOP_SENT_LENGHT:
                    bag.append([])
                    seqs[len(seqs)-1] += 1
                bag[len(bag) - 1].append(k)

            if len(bag) == SENTENCES:
                break

        for j in range(len(bag)):
            for k in range(len(bag[j])):
                emb = dicc.get(bag[j][k])
                if emb is not None:
                    to_encode[pi][j][k] += emb
                    aproached_words[len(aproached_words) - 1] += 1
                else:
                    losed_words[len(losed_words) - 1] += 1
                    to_encode[pi][j][k] += UNK_TOKEN

            for k in range(len(bag[j]), TOP_SENT_LENGHT, 1):
                to_encode[pi][j][k] += PAD_TOKEN
        pi += 1

    return  to_encode, seqs, ages, topic, sex, embeddings_matrix, ids

def lematized_for_topic(embeddings_matrix, dicc, DATA_PATH, mode, stop_words):

    def load_data_LEMA(data_path):
        postxuser = []
        print(data_path)
        XTree = open(data_path, mode='rU')
        users = []
        band = False
        for line in XTree:
            usr = line.split(' ')

            if usr[0] == "<user":
                users.append({'post': '', 'id': usr[1][4:len(usr[1]) - 1], 'age': usr[3][len("topic"):len(usr[3]) - 1],
                              'gender': usr[4][len("gender= "):len(usr[4]) - 3],
                              'topic': usr[2][len("topic= "):len(usr[2]) - 1]})
                postxuser.append(0)
            if band:
                if line[:len(line) - 1] == '':
                    band = False
                    continue
                else:
                    tmp = line[:len(line) - 1]
                    for k in emoji.keys():
                        while True:
                            z = str.find(tmp, k)
                            if z == -1:
                                break
                            tmp1 = tmp[z + len(k):]
                            tmp = tmp[:z] + '< emoticon > ' + emoji[k] + ' '
                            tmp += tmp1
                    users[len(users) - 1]['post'] += tmp

                if users[len(users) - 1]['post'][len(users[len(users) - 1]['post']) - 1] not in {'?', '!', '.', '...'}:
                    users[len(users) - 1]['post'] += '.'
                users[len(users) - 1]['post'] += ' '
                band = False

            if usr[0][:len(usr[0]) - 1] == "<post>":
                band = True
                postxuser[len(postxuser) - 1] += 1
        return users, postxuser


    def Preprocess_LEMA(users, dicc, stop_words):
        print('Freeling Tokenizing ', '0 %\r', end="")
        if "FREELINGDIR" not in os.environ:
            os.environ["FREELINGDIR"] = "/usr/local"
        DATA = os.environ["FREELINGDIR"] + "/share/freeling/"

        pyfreeling.util_init_locale("default")
        LANG = "it"
        op = pyfreeling.maco_options(LANG)
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
        cont = 0
        for i in range(len(users)):

            done += 1
            z = done / top
            z *= 100
            z = int(z)
            if z - perc >= 1:
                perc = z
                print('Freeling Tokenizing ', str(perc) + ' %\r', end="")

            x = users[i]['post']
            l = tk.tokenize(x)
            #         sid=sp.open_session();
            ls = sp.split(l)

            ls = mf.analyze(ls)
            ls = tg.analyze(ls)
            ls = sen.analyze(ls)

            listsent = []
            for s in ls:
                cont  += 1
                ora = s.get_words()
                tmp = []
                for k in range(len(ora)):
                    x = ora[k].get_lemma().lower()
                    if ora[k].get_tag() != 'W' and ora[k].get_tag() != 'Z' and ora[k].get_tag()[0] != 'F' and stop_words.count(x) == 0 and dicc.get(x) is not None:
                        listsent.append(x)
            users[i]['post'] = np.array(listsent)

        print('Freeling Tokenizing ok        ')
        return users

    profiles, postxuser = load_data_LEMA(DATA_PATH) 
    profiles = Preprocess_LEMA(profiles, dicc, stop_words)


    pi = 0
    PAD_TOKEN = len(dicc)
    UNK_TOKEN = len(dicc)+1
    aproached_words = []
    losed_words = []
    TOP_SENT_LENGHT = 400

    test = np.zeros((len(profiles), TOP_SENT_LENGHT))
    for i in range(len(profiles)):


        aproached_words.append(0)
        losed_words.append(0)
        for j in range(len(profiles[i]['post'])):

            if j >= TOP_SENT_LENGHT:
                break

            emb = dicc.get(profiles[i]['post'][j])
            if emb is not None:
                test[pi][j] += emb
                aproached_words[len(aproached_words) - 1] += 1
            else:
                losed_words[len(losed_words) - 1] += 1
                test[pi][j] += UNK_TOKEN

        for k in range(len(profiles[i]['post']), TOP_SENT_LENGHT, 1):
            test[pi][k] += PAD_TOKEN
        pi += 1
    # np.save('lematized_profiels_sequences_topic', test)
    labels = np.zeros((test.shape[0],))
    index = np.array(np.zeros((test.shape[0],)), dtype=np.str_)
    for i in range(len(labels)):
        if mode == 'train':
            labels[i] = TOPIC_CLASSES.index(profiles[i]['topic'])
        index[i] = profiles[i]['id']

    return test, labels, index

def load_for_age(embeddings_matrix, dicc, DATA_PATH, labeltype, mode):

    def load_data_AGE(data_path):
        postxuser = []
        print(data_path)
        XTree = open(data_path, mode='rU')
        users = []
        band = False
        for line in XTree:
            usr = line.split(' ')

            if usr[0] == "<user":
                users.append({'post': '', 'id': int(usr[1][4:len(usr[1]) - 1]), 'age': usr[3][len("topic"):len(usr[3]) - 1],
                              'gender': usr[4][len("gender= "):len(usr[4]) - 3],
                              'topic': usr[2][len("topic= "):len(usr[2]) - 1]})
                postxuser.append(0)
            if band:
                if line[:len(line) - 1] == '':
                    band = False
                    continue
                else:
                    tmp = line[:len(line) - 1]
                    for k in emoji.keys():
                        while True:
                            z = str.find(tmp, k)
                            if z == -1:
                                break
                            tmp1 = tmp[z + len(k):]
                            tmp = tmp[:z] + '< emoticon > '
                            tmp += tmp1
                    users[len(users) - 1]['post'] += tmp

                if users[len(users) - 1]['post'][len(users[len(users) - 1]['post']) - 1] not in {'?', '!', '.', '...'}:
                    users[len(users) - 1]['post'] += '.'
                users[len(users) - 1]['post'] += ' '
                band = False

            if usr[0][:len(usr[0]) - 1] == "<post>":
                band = True
                postxuser[len(postxuser) - 1] += 1
        return users, postxuser

    def Preprocess_AGE(users, dicc):
        print('Freeling Tokenizing ', '0 %\r', end="")
        if "FREELINGDIR" not in os.environ:
            os.environ["FREELINGDIR"] = "/usr/local"
        DATA = os.environ["FREELINGDIR"] + "/share/freeling/";

        pyfreeling.util_init_locale("default");
        LANG = "it";
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
        for i in range(len(users)):

            done += 1
            z = done / top
            z *= 100
            z = int(z)
            if z - perc >= 1:
                perc = z
                print('Freeling Tokenizing ', str(perc) + ' %\r', end="")

            x = users[i]['post']
            l = tk.tokenize(x);
            #         sid=sp.open_session();
            ls = sp.split(l);

            ls = mf.analyze(ls);
            ls = tg.analyze(ls);
            ls = sen.analyze(ls);

            listsent = []
            for s in ls:
                cont  += 1
                ora = s.get_words()
                tmp = []
                for k in range(len(ora)):
                    x = ora[k].get_form().lower()
                    if ora[k].get_tag() == 'W':
                        listsent.append('data')
                    elif ora[k].get_tag() == 'Z':
                        listsent.append('numero')
                    elif dicc.get(x) is not None:
                        listsent.append(x)

            users[i]['post'] = np.array(listsent)

        print('Freeling Tokenizing ok        ')
        return users

    profiles, postxuser = load_data_AGE(DATA_PATH) 
    profiles = Preprocess_AGE(profiles, dicc)

    pi = 0
    PAD_TOKEN = len(dicc)
    UNK_TOKEN = len(dicc)+1
    aproached_words = []
    losed_words = []
    TOP_SENT_LENGHT = 1050

    test = np.zeros((len(profiles), TOP_SENT_LENGHT))
    for i in range(len(profiles)):

        aproached_words.append(0)
        losed_words.append(0)
        for j in range(len(profiles[i]['post'])):

            if j >= TOP_SENT_LENGHT:
                break

            emb = dicc.get(profiles[i]['post'][j])
            if emb is not None:
                test[pi][j] += emb
                aproached_words[len(aproached_words) - 1] += 1
            else:
                losed_words[len(losed_words) - 1] += 1
                test[pi][j] += UNK_TOKEN

        for k in range(len(profiles[i]['post']), TOP_SENT_LENGHT, 1):
            test[pi][k] += PAD_TOKEN
        pi += 1
        # np.save('data_for_age', test)
    labels = np.zeros((test.shape[0],))
    index = np.array(np.zeros((test.shape[0],)), dtype=np.str_)
    for i in range(len(labels)):
        index[i] = profiles[i]['id']

    if labeltype == 'sex' and mode == 'train':
        for i in range(len(labels)):
            if profiles[i]['gender'] == 'F':
                labels[i] = 1

    if labeltype == 'age' and mode == 'train':
        for i in range(len(labels)):
            labels[i] = AGE_CLASSES.index(profiles[i]['age'])
    return test, labels, index