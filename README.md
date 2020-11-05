## EVALITA_TAG_it

This Project constist on a neural network model used for participating in the TAG-it Author Profiling task from EVALITA 2020, on its different subtasks. Aiming to predict age and gender of blogs users from their posts, as the topic they wrote about. 
It combines learned representations by RNN at word and sentence levels, Transformer Neural Net, specifically BERT model and hand-crafted stylistic features. 
All these representations are mixed and fed into fully connected layer from a fedforward neural network in order to make predictions for addressed subtasks.

The Models description is available [here](https://www.google.com/).

For this code be functional is needed:
- Python 3.8
- tensorflow 2.0
- Keras 2.4.3
- Freeling 4.1 and python API
- Italian Word Embedding avalilable [here](https://fasttext.cc/docs/en/crawl-vectors.html)

### Steps for using the model
  - Once downloaded the word embedding file `(wiki-it.vec)` it must be placed on ``data`` folder. This folder contains the used BERT model as the weights of its training.
  - Download the weights of the [BERT model](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip) and place it on `data' folder.
  - [Train the models](#Training-models-of-the-ensembler).
  - [Make the predictions over the test files](#Making-Predictions)

### Training models of the ensembler

The code of models for predicting each task is locatend on Ensemble floder, also there is a file train.py which once run save the weights learned with the provided training data.
So the first step for use this classifier is run on the command line:

```shell
 python ./Ensemble/train.py
```
The training files are located on ``data`` folder and are the one provided by the contest organizers. If you want to chage the trainning file change the `source` variable on this `train.py` file.

### Making Predictions

For making predictions run:

```shell
 python main.py
```
 You should provide the test files by `-dp` option. Inside the `test_data` folder is the test data provided by the organizers.
 
 ### Data Format
 The datasets are composed by texts written by multiple users, with possibly multiple posts per user.
 The data is distributed in the form of one XML-like file per genre with one sample per elements, and attributes specifying an id, the topic,  the gender `male|female`, and the age range `[0,19], [20,29], [30-39], [40-49], [50-100]`. This is a sample:
 
 ``` xml
 
<doc id="3046" topic="orologi" age="30-39" gender="male" >
  <post>
    Per quale motivo oggi, il mondo dell'orologeria è così importante per voi? 
  </post>
  <post>
    Cosa vi ha spinto a rendervi appassionati così bramosi?
  </post>
</doc>
 ```

