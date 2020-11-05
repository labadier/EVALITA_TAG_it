import os
import sys

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Make Predictions')

    parser.add_argument('-st', metavar='-subtask', required=True, help='Subtask id')
    parser.add_argument('-dp', metavar='-datapath', required=True, help='Path of Data for predict')
    args = parser.parse_args()
    subtask = args.st  
    data_source = args.dp

    def get_age(age, indexes):
        with open('./output/output_age.txt', mode='rU') as file:
            for i in file:
                index, content = i.split(maxsplit=1)
                indexes.add(index)
                age[index] = content[:-1]
        return age, indexes

    def get_topic(topic, indexes):
        with open('./output/output_topic.txt') as file:
            for i in file:
                index, content = i.split(' ')
                indexes.add(index)
                topic[index] = content[:-1]
        return topic, indexes

    def get_gender(sex, indexes):
        with open('./output/output_sex.txt') as file:
            for i in file:
                index, content = i.split(' ')
                indexes.add(index)
                sex[index] = content[:-1]
        return sex, indexes

    os.system('mkdir output')
    file = "test_task" + subtask + ".txt"
    file = os.path.join(data_source, file)
    original_path = os.getcwd()
    os.system("python bertTL.py -d " +  file)
    os.system("python feature_extract.py -d " + file)
    os.system("python feature_extract_IG_ITFIDF.py -d " + file)
    os.chdir('./Ensembler/Classifiers/')

    # print(file)
    os.system("python Sentence_Encoder.py -d " + file)
    # os.chdir(original_path)
    # exit()

    age = {}
    topic = {}
    sex = {}
    indexes = set()

    if subtask == '1':
        os.system("python Classifier_Topic.py -d " + file)
        os.chdir(original_path)
        topic, indexes = get_topic(topic, indexes)
        os.chdir('./Ensembler/Classifiers/')
    
    if subtask == '2b' or subtask == '1':
        os.system("python Classifier_Ages.py -d " + file)
        os.chdir(original_path)
        age, indexes = get_age(age, indexes)
        os.chdir('./Ensembler/Classifiers/')
   
    if subtask == '1' or subtask == '2a':
        os.system("python Classifier_Sex.py -d " + file)
        os.chdir(original_path)
        sex, indexes = get_gender(sex, indexes)
        os.chdir('./Ensembler/Classifiers/')

    os.chdir(original_path)
    path_ans = './output/UOBIT_' + subtask + '_3'
    if subtask != '1':
        for i in indexes:
            topic[i] = 'X'
            if subtask == '2b':
                sex[i] = 'X'
            else: age[i] = 'X'

    with open(path_ans, 'w') as file:
        indexes = [int(i) for i in indexes]
        indexes.sort()
        indexes = [str(i) for i in indexes]
        for i in indexes:
            file.write(i + "\t" + sex[i] + '\t' + age[i] + '\t' + topic[i] + '\n')
