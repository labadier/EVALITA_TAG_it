import os
import sys

# l = ["RNNS-RNNW-Feat-B", "RNNS-RNNW-B", "RNNS-Feat-B", "RNNW-Feat-B"]
l = ["RNNS-RNNW-B"]
source = "./data/training.txt"
sourcemodels = ' -dp .' + source

os.system("python model_sentenc.py" + sourcemodels )
os.system("python model_bertTL.py" + sourcemodels )

original_path = os.getcwd()

os.chdir('../')
os.system("python bertTL.py -d " + source)
os.system("python feature_extract.py -d " + source)
os.system("python feature_extract_IG_ITFIDF.py -d " + source)
os.chdir(os.path.join(original_path, 'Classifiers'))
os.system("python Sentence_Encoder.py -d ../." + source)
os.chdir(original_path)


for i in l:
    os.system("python model_age.py -md " + i + sourcemodels )
    os.system("python model_sex.py -md " + i + sourcemodels )
    os.system("python model_topic.py -md " + i + sourcemodels )
