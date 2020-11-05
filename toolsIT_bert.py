import numpy as np
import pyfreeling
import os
import json

emoji = {':-)': 'sorridi', '(-:': 'sorridi', ':)': 'sorridi', '(:': 'sorridi', '=]': 'sorridi', ':]': 'sorridi',
         ':d': 'ridere', '8d': 'ridere', 'xd': 'ridere', '=d': 'ridere', ':-d': 'ridere',
         ':D': 'ridere', '8D': 'ridere', 'xD': 'ridere', '=D': 'ridere', ':-D': 'ridere',
         'x-d': 'ridere', ':(': 'triste', '):': 'triste', ':c': 'triste', ':[': 'triste', ':\'(': 'pianto',
         ':,( ': 'pianto', 'x-D': 'ridere', ':"(': 'pianto', ':((': 'pianto', ':|': 'neutrale',
         '=_=': 'neutrale', '-_-': 'neutrale', ':-\\': 'neutrale', ':O': 'neutrale', ':-!': 'neutrale'}

AGE_CLASSES = ['0-19', '20-29', '30-39', '40-49', '50-100']
TOPIC_CLASSES = ['ANIME', 'AUTO-MOTO', 'BIKES', 'CELEBRITIES', 'ENTERTAINMENT', 'MEDICINE-AESTHETICS',
                 'METAL-DETECTING', 'NATURE', 'SMOKE', 'SPORTS', 'TECHNOLOGY']


def load_stylistic_features(STYLE_PATH):
    with open(STYLE_PATH) as F:
        data = json.load(F) 
    STYF = [np.fromstring(data['training']['txt'][str(i+1)]['vec'], 'f', sep=',') for i in range(len(data['training']['txt']))]
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
            users.append({'post': '', 'id': int(usr[1][4:len(usr[1]) - 1]), 'age': usr[3][len("topic"):len(usr[3]) - 1],
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
            band = False

        if usr[0][:len(usr[0]) - 1] == "<post>":
            band = True
            postxuser[len(postxuser) - 1] += 1
    return users, postxuser


def Preprocess(users):
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
                          True, True, False, True)  # default: all created submodules are used

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
                if ora[k].get_tag() == 'W':
                    tmp.append( 'data' )
                elif ora[k].get_tag() == 'Z':
                    tmp.append( 'numero' )
                else:
                    tmp.append(ora[k].get_form().lower())
#                 print(ora[k].get_form().lower(), ora[k].get_tag())
            listsent.append(tmp)
        users[i]['post'] = np.array(listsent)

    print('Freeling Tokenizing ok        ')
    return users
    
def load_for_age(DATA_PATH, mode):
    

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

    def Preprocess_AGE(users):
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
                              True, True, False, True)  # default: all created submodules are used

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
                    x = ora[k].get_form().lower()
                    if ora[k].get_tag() == 'W':
                        listsent.append('data')
                    elif ora[k].get_tag() == 'Z':
                        listsent.append('numero')
                    else: listsent.append(x)

            users[i]['post'] = np.array(listsent)

        print('Freeling Tokenizing ok        ')
        return users

    profiles, postxuser = load_data_AGE(DATA_PATH)
    profiles = Preprocess_AGE(profiles)

    TOP_SENT_LENGHT = 348
    to_encode = np.array(np.zeros((len(profiles), TOP_SENT_LENGHT)), dtype=np.str_)
    for i in range(len(profiles)):
        for j in range(int(min(TOP_SENT_LENGHT, len(profiles[i]['post'])))):
            to_encode[i][j] = profiles[i]['post'][j]
    
    labels = np.zeros((len(to_encode), 3))
    if mode != 'test':
        for i in range(len(profiles)):
            labels[i][0] = AGE_CLASSES.index(profiles[i]['age'])
            labels[i][1] = TOPIC_CLASSES.index(profiles[i]['topic'])
            if profiles[i]['gender'] == 'F':
                labels[i][2] = 1
            
    return to_encode, labels
