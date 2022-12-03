import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec, FastText
# import glove
# from glove import Corpus

import collections
import gc 

import keras
from keras import backend as K
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Input, concatenate, merge, Activation, Concatenate, LSTM, GRU
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, BatchNormalization, GRU, Convolution1D, LSTM
from keras.layers import UpSampling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,MaxPool1D, merge

# from keras.optimizers import Adam
from keras.optimizers import adam_v2

from keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau
from keras.utils import np_utils
# from keras.backend.tensorflow_backend import set_session, clear_session, get_session
from tensorflow.python.keras.backend import set_session, clear_session, get_session
import tensorflow as tf


from sklearn.utils import class_weight
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

import warnings
warnings.filterwarnings('ignore')


# Reset Keras Session
def reset_keras(model):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del model # this is from global space - change this as you need
    except:
        pass

    gc.collect() # if it's done something you should see a number being outputted
    
# trans x data into input format
# def create_dataset(dict_of_ner):
def create_dataset(dict_of_ner, embed_name):
    siz = 0
    if embed_name == "concat":
        siz = 400
    elif embed_name == "fasttext":
        siz = 300
    else:
        siz = 100
    temp_data = []
    for k, v in sorted(dict_of_ner.items()):
        temp = []
        for embed in v:
            # temp.append(embed)
            try:
                sss = len(embed)
                # print("sss", sss)
            except:
                # continue
                embed = [0.0] * siz
            temp.append(embed)
        # temp_data.append(np.mean(temp, axis = 0)) 
        mmm = np.mean(temp, axis = 0)
        try:
            lll = len(mmm)
            # if lll == 100:
            if lll == siz:
                temp_data.append(np.mean(temp, axis = 0))    
        except:
            pass
        
    return np.asarray(temp_data)

# change probability into binary value of 0 or 1
def make_prediction_multi_avg(model, test_data):
    probs = model.predict(test_data)
    y_pred = [1 if i>=0.5 else 0 for i in probs]
    return probs, y_pred

#  save results into files
def save_scores_multi_avg(predictions, probs, ground_truth, 
                          
                          embed_name, problem_type, iteration, hidden_unit_size,
                          
                          sequence_name, type_of_ner):
    
    auc = roc_auc_score(ground_truth, probs)
    auprc = average_precision_score(ground_truth, probs)
    acc   = accuracy_score(ground_truth, predictions)
    F1    = f1_score(ground_truth, predictions)
    bas    = balanced_accuracy_score(ground_truth, predictions)
    
    result_dict = {}    
    result_dict['auc'] = auc
    result_dict['auprc'] = auprc
    result_dict['acc'] = acc
    result_dict['F1'] = F1
    
    result_path = "results/"
    file_name = str(sequence_name)+"-"+str(hidden_unit_size)+"-"+embed_name
    file_name = file_name +"-"+problem_type+"-"+str(iteration)+"-"+type_of_ner+"-avg-.p"
    pd.to_pickle(result_dict, os.path.join(result_path, file_name))

    # print(auc, auprc, acc, F1)
    print ("AUC: ", auc, "AUPRC: ", auprc, "F1: ", F1, "Acc: ", acc, "Bal Acc: ", bas)
    
# create NLP based prediction model
def avg_ner_model(layer_name, number_of_unit, embedding_name):

    # if embedding_name == "concat":
    #     input_dimension = 200
    # else:
    #     input_dimension = 100
        
    if embedding_name == "concat":
        input_dimension = 400
    elif embedding_name == "fasttext":
        input_dimension = 300
    else:
        input_dimension = 100

    sequence_input = Input(shape=(24,104))

    input_avg = Input(shape=(input_dimension, ), name = "avg")        
    
    if layer_name == "GRU":
        x = GRU(number_of_unit)(sequence_input)
    elif layer_name == "LSTM":
        x = LSTM(number_of_unit)(sequence_input)

    x = keras.layers.Concatenate()([x, input_avg])

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    preds = Dense(1, activation='sigmoid',use_bias=False,
                         kernel_initializer=tf.initializers.GlorotUniform(), 
                  kernel_regularizer=regularizers.L2(0.01))(x)

    opt = adam_v2.Adam(lr=0.001, decay = 0.01)

    model = Model(inputs=[sequence_input, input_avg], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
    
    return model

def mean(a):
    return sum(a) / len(a)

# import x and y data
type_of_ner = "new"

x_train_lstm = pd.read_pickle("data/"+type_of_ner+"_x_train.pkl")
x_dev_lstm = pd.read_pickle("data/"+type_of_ner+"_x_dev.pkl")
x_test_lstm = pd.read_pickle("data/"+type_of_ner+"_x_test.pkl")

y_train = pd.read_pickle("data/"+type_of_ner+"_y_train.pkl")
y_dev = pd.read_pickle("data/"+type_of_ner+"_y_dev.pkl")
y_test = pd.read_pickle("data/"+type_of_ner+"_y_test.pkl")

ner_word2vec = pd.read_pickle("data/"+type_of_ner+"_ner_word2vec_limited_dict.pkl")
ner_fasttext = pd.read_pickle("data/"+type_of_ner+"_ner_fasttext_limited_dict.pkl")
ner_concat = pd.read_pickle("data/"+type_of_ner+"_ner_combined_limited_dict.pkl")

train_ids = pd.read_pickle("data/"+type_of_ner+"_train_ids.pkl")
dev_ids = pd.read_pickle("data/"+type_of_ner+"_dev_ids.pkl")
test_ids = pd.read_pickle("data/"+type_of_ner+"_test_ids.pkl")

embedding_types = ['word2vec', 'fasttext', 'concat']
embedding_dict = [ner_word2vec, ner_fasttext, ner_concat]

target_problems = ['mort_hosp', 'mort_icu', 'los_3', 'los_7']


num_epoch = 100
# num_epoch = 1
# num_epoch = 3
model_patience = 5
monitor_criteria = 'val_loss'
batch_size = 64
iter_num = 4
# iter_num = 1
unit_sizes = [128, 256]
# unit_sizes = [128]

xxx = []
layers = ["LSTM", "GRU"]
# layers = ["GRU"]
for each_layer in layers:
    print ("Layer: ", each_layer)
    for each_unit_size in unit_sizes:
        print ("Hidden unit: ", each_unit_size)

        for embed_dict, embed_name in zip(embedding_dict, embedding_types):    
            print ("Embedding: ", embed_name)
            print("=============================")

            # Create input embedding data

            temp_train_ner = {}
            temp_dev_ner = {}
            temp_test_ner = {}
            if embed_name == "word2vec":
                temp_train_ner = dict((k, embed_dict[k]) for k in train_ids)
                temp_dev_ner = dict((k, embed_dict[k]) for k in dev_ids)
                temp_test_ner = dict((k, embed_dict[k]) for k in test_ids)
            elif embed_name == "fasttext":
                np300 = np.array([0.0] * 300)
                bbb = []
                for k in train_ids:
                    if k in embed_dict:
                        bbb.append((k, embed_dict[k]))
                    else:
                        bbb.append((k, [np300]))
                        # pass
                temp_train_ner = dict(bbb)
                bbb = []
                for k in dev_ids:
                    if k in embed_dict:
                        bbb.append((k, embed_dict[k]))
                    else:
                        bbb.append((k, [np300]))
                        # pass
                temp_dev_ner = dict(bbb)
                bbb = []
                for k in test_ids:
                    if k in embed_dict:
                        bbb.append((k, embed_dict[k]))
                    else:
                        bbb.append((k, [np300]))
                        # pass
                temp_test_ner = dict(bbb)
                
            elif embed_name == "concat":
                np100 = np.array([0.0] * 400)
                bbb = []
                for k in train_ids:
                    if k in embed_dict:
                        # xxx.append(len(embed_dict[k][0]))
                        bbb.append((k, embed_dict[k]))
                    else:
                        bbb.append((k, [np100]))
                temp_train_ner = dict(bbb)
                bbb = []
                for k in dev_ids:
                    if k in embed_dict:
                        bbb.append((k, embed_dict[k]))
                    else:
                        bbb.append((k, [np100]))
                temp_dev_ner = dict(bbb)
                bbb = []
                for k in test_ids:
                    if k in embed_dict:
                        bbb.append((k, embed_dict[k]))
                    else:
                        bbb.append((k, [np100]))
                temp_test_ner = dict(bbb)

            x_train_ner = create_dataset(temp_train_ner, embed_name)
            x_dev_ner = create_dataset(temp_dev_ner, embed_name)
            x_test_ner = create_dataset(temp_test_ner, embed_name)

            for iteration in range(1, iter_num):
                print ("Iteration number: ", iteration)

                for each_problem in target_problems:
                    print ("Problem type: ", each_problem)
                    print ("__________________")

                    # Stop training when a monitored metric has stopped improving
                    early_stopping_monitor = EarlyStopping(monitor=monitor_criteria, patience=model_patience)
                    best_model_name = "avg-"+str(embed_name)+"-"+str(each_problem)+"-"+"best_model.hdf5"
                    # Callback to save the Keras model or model weights at some frequency.
                    checkpoint = ModelCheckpoint(best_model_name, monitor='val_loss', verbose=1,
                        save_best_only=True, mode='min', period=1)


                    callbacks = [early_stopping_monitor, checkpoint]

                    # create prediction model
                    model = avg_ner_model(each_layer, each_unit_size, embed_name)

                    model.fit([x_train_lstm, x_train_ner], y_train[each_problem], epochs=num_epoch, verbose=1, 
                              validation_data=([x_dev_lstm, x_dev_ner], y_dev[each_problem]), callbacks=callbacks, 
                              batch_size=batch_size )

                    model.load_weights(best_model_name)

                    # make prediction
                    probs, predictions = make_prediction_multi_avg(model, [x_test_lstm, x_test_ner])
                    
                    save_scores_multi_avg(predictions, probs, y_test[each_problem], 
                               embed_name, each_problem, iteration, each_unit_size, 
                               each_layer, type_of_ner)
                    
                    reset_keras(model)
                    del model
                    clear_session()
                    gc.collect()