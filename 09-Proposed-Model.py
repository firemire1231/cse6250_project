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
# from keras.layers import UpSampling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,MaxPool1D, merge
from keras.layers import UpSampling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D,MaxPool1D, concatenate

# from keras.optimizers import Adam
from keras.optimizers import adam_v2

from keras.callbacks import EarlyStopping, ModelCheckpoint, History, ReduceLROnPlateau
from keras.utils import np_utils
# from keras.backend.tensorflow_backend import set_session, clear_session, get_session
from tensorflow.python.keras.backend import set_session, clear_session, get_session
import tensorflow as tf


from sklearn.utils import class_weight
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

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


# change probability into binary value of 0 or 1
def make_prediction_cnn(model, test_data):
    probs = model.predict(test_data)
    y_pred = [1 if i>=0.5 else 0 for i in probs]
    return probs, y_pred

# change probability into binary value of 0 or 1
def make_prediction_cnn2(model, test_data, thresh=0.5):
    probs = model.predict(test_data)
    y_pred = [1 if i>=thresh else 0 for i in probs]
    return probs, y_pred

#  save results into files
def save_scores_cnn(predictions, probs, ground_truth, 
                          
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
    result_dict['bas'] = bas

    result_path = "results/cnn/"
    file_name = str(sequence_name)+"-"+str(hidden_unit_size)+"-"+embed_name
    file_name = file_name +"-"+problem_type+"-"+str(iteration)+"-"+type_of_ner+"-cnn-.p"
    pd.to_pickle(result_dict, os.path.join(result_path, file_name))

    # print(auc, auprc, acc, F1)
  
#  print the results
def print_scores_cnn(predictions, probs, ground_truth, model_name, problem_type, iteration, hidden_unit_size):
    auc = roc_auc_score(ground_truth, probs)
    auprc = average_precision_score(ground_truth, probs)
    acc   = accuracy_score(ground_truth, predictions)
    F1    = f1_score(ground_truth, predictions)
    bas    = balanced_accuracy_score(ground_truth, predictions)
    
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    recall = tp / (tp+fn)
    specificity = tn / (tn+fp)

    # print ("AUC: ", auc, "AUPRC: ", auprc, "F1: ", F1, "Acc: ", acc, "Bal Acc: ", bas, "Recall: ", recall, "Specificity: ", specificity)
    print ("AUC: ", auc, "AUPRC: ", auprc, "F1: ", F1, "Acc: ", acc, "Bal Acc: ", bas)
    
# convert into input vectors
def get_subvector_data(size, embed_name, data):
    # if embed_name == "concat":
    #     vector_size = 200
    # else:
    #     vector_size = 100
    if embed_name == "concat":
        vector_size = 400
    elif embed_name == "fasttext":
        vector_size = 300
    else:
        vector_size = 100
        
    x_data = {}
    for k, v in data.items():
        number_of_additional_vector = len(v) - size
        vector = []
        for i in v:
            vector.append(i)
        if number_of_additional_vector < 0: 
            number_of_additional_vector = np.abs(number_of_additional_vector)

            temp = vector[:size]
            for i in range(0, number_of_additional_vector):
                temp.append(np.zeros(vector_size))
            x_data[k] = np.asarray(temp)
        else:
            x_data[k] = np.asarray(vector[:size])

    return x_data


# create CNN + RNN prediction model
def proposedmodel(layer_name, number_of_unit, embedding_name, ner_limit, num_filter):
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

    input_img = Input(shape=(ner_limit, input_dimension), name = "cnn_input")

    convs = []
    filter_sizes = [2,3,4]



    text_conv1d = Conv1D(filters=num_filter, kernel_size=3, 
                 padding = 'valid', strides = 1, dilation_rate=1, activation='relu', 
                        #  kernel_initializer=tf.contrib.layers.xavier_initializer() )(input_img)
                         kernel_initializer=tf.initializers.GlorotUniform())(input_img)
    
    text_conv1d = Conv1D(filters=num_filter*2, kernel_size=3, 
                 padding = 'valid', strides = 1, dilation_rate=1, activation='relu',
                        # kernel_initializer=tf.contrib.layers.xavier_initializer())(text_conv1d)   
                        kernel_initializer=tf.initializers.GlorotUniform())(text_conv1d)   
    
    text_conv1d = Conv1D(filters=num_filter*3, kernel_size=3, 
                 padding = 'valid', strides = 1, dilation_rate=1, activation='relu',
                        # kernel_initializer=tf.contrib.layers.xavier_initializer())(text_conv1d)   
                        kernel_initializer=tf.initializers.GlorotUniform())(text_conv1d)   


    
    #concat_conv = keras.layers.Concatenate()([text_conv1d, text_conv1d_2, text_conv1d_3])
    text_embeddings = GlobalMaxPooling1D()(text_conv1d)
    #text_embeddings = Dense(128, activation="relu")(text_embeddings)
    
    if layer_name == "GRU":
        x = GRU(number_of_unit)(sequence_input)
    elif layer_name == "LSTM":
        x = LSTM(number_of_unit)(sequence_input)


    concatenated = concatenate([x, text_embeddings],axis=1)

    # concatenated = Dense(512, activation='relu')(concatenated)
    concatenated = Dense(512, activation='relu')(concatenated)
    concatenated = Dropout(0.2)(concatenated)

    
    preds = Dense(1, activation='sigmoid',use_bias=False,
                         kernel_initializer=tf.initializers.GlorotUniform(), 
                  kernel_regularizer=regularizers.L2(0.01))(concatenated)
                #   kernel_regularizer=regularizers.L1(0.1))(concatenated)
                

    opt = adam_v2.Adam(lr=1e-3, decay = 0.01)

    model = Model(inputs=[sequence_input, input_img], outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])
                #   metrics=[tf.keras.metrics.TrueNegatives()])
                # metrics=[tf.keras.metrics.SpecificityAtSensitivity(0.50)])
                # metrics=[tf.keras.metrics.AUC()])
    
    return model

# convert into input dataset
def create_dataset(dict_of_ner, embed_name, num_dim):
    num_dim = 64
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
        # print("len(v)", len(v))
        i = 0
        for embed in v:
            if i == num_dim: break
            # temp.append(embed)
            try:
                sss = len(embed)
                    # print("sss", sss)
            except:
                    # continue
                embed = [0.0] * siz
            temp.append(embed)
            # temp_data.append(np.mean(temp, axis = 0)) 
        # mmm = np.mean(temp, axis = 0)
            i += 1
        for i in range(num_dim - len(temp)):
            temp.append([0.0] * siz)
        temp_data.append(np.array(temp))
    
    return np.asarray(temp_data)


embedding_types = ['word2vec', 'fasttext', 'concat']
embedding_dict = [ner_word2vec, ner_fasttext, ner_concat]
# embedding_types = ['word2vec']
# embedding_dict = [ner_word2vec]

target_problems = ['mort_hosp', 'mort_icu', 'los_3', 'los_7']
# target_problems = ['los_3']

num_epoch = 100
# num_epoch = 3
# num_epoch = 30
# num_epoch = 1
model_patience = 5
# model_patience = 10
monitor_criteria = 'val_loss'
# monitor_criteria = 'val_specificity_at_sensitivity'
# monitor_criteria = 'val_true_negatives'
# monitor_criteria = 'val_auc'
# monitor_criteria = 'val_acc'
batch_size = 64

filter_number = 32
ner_representation_limit = 64
activation_func = "relu"

sequence_model = "GRU"
sequence_hidden_unit = 256

ZZZ = 0
DEBUG_x_train_ner = 0
TEMP_train_ner = 0

# maxiter = 11
# maxiter = 2
maxiter = 4
# for sequence_model in ["GRU", "LSTM"]:
for sequence_model in ["GRU"]:
    print ("Layer: ", sequence_model)
    print("=============================")
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

        x_train_ner = create_dataset(temp_train_ner, embed_name, ner_representation_limit)
        x_dev_ner = create_dataset(temp_dev_ner, embed_name, ner_representation_limit)
        x_test_ner = create_dataset(temp_test_ner, embed_name, ner_representation_limit)
        
        TEMP_train_ner = temp_train_ner
        
        ZZZ = x_train_ner
        for iteration in range(1,maxiter):
            print ("Iteration number: ", iteration)
        
            for each_problem in target_problems:
                print ("Problem type: ", each_problem)
                print ("__________________")
                
                
                # Stop training when a monitored metric has stopped improving
                early_stopping_monitor = EarlyStopping(monitor=monitor_criteria, patience=model_patience)
                
                best_model_name = str(ner_representation_limit)+"-basiccnn1d-"+str(embed_name)+"-"+str(each_problem)+"-"+"best_model.hdf5"
                
                # Callback to save the Keras model or model weights at some frequency.
                checkpoint = ModelCheckpoint(best_model_name, monitor=monitor_criteria, verbose=1,
                    # save_best_only=True, mode='min')
                    save_best_only=True, mode='max')
                
                reduce_lr = ReduceLROnPlateau(monitor=monitor_criteria, factor=0.2,
                                # patience=2, min_lr=0.00001, epsilon=1e-4, mode='min')
                                patience=2, max_lr=0.00001, epsilon=1e-4, mode='max')
                

                callbacks = [early_stopping_monitor, checkpoint, reduce_lr]
                
                DEBUG_x_train_ner = x_train_ner
                #model = textCNN(sequence_model, sequence_hidden_unit, embed_name, ner_representation_limit)
                
                # create prediction model
                model = proposedmodel(sequence_model, sequence_hidden_unit, 
                                embed_name, ner_representation_limit,filter_number)
                model.fit([x_train_lstm, x_train_ner], y_train[each_problem], epochs=num_epoch, verbose=1, 
                        validation_data=([x_dev_lstm, x_dev_ner], y_dev[each_problem]), callbacks=callbacks, batch_size=batch_size)
                
                thresh = 0.45
                probs, predictions = make_prediction_cnn2(model, [x_test_lstm, x_test_ner], thresh)
                print ("Threshold = ", thresh)
                print_scores_cnn(predictions, probs, y_test[each_problem], embed_name, each_problem, iteration, sequence_hidden_unit)
                thresh = 0.55
                probs, predictions = make_prediction_cnn2(model, [x_test_lstm, x_test_ner], thresh)
                print ("Threshold = ", thresh)
                print_scores_cnn(predictions, probs, y_test[each_problem], embed_name, each_problem, iteration, sequence_hidden_unit)
                
                print ("Threshold = 0.5")
                probs, predictions = make_prediction_cnn(model, [x_test_lstm, x_test_ner])
                print_scores_cnn(predictions, probs, y_test[each_problem], embed_name, each_problem, iteration, sequence_hidden_unit)
                
                model.load_weights(best_model_name)
                        
                # make prediction
                probs, predictions = make_prediction_cnn(model, [x_test_lstm, x_test_ner])
                save_scores_cnn(predictions, probs, y_test[each_problem], embed_name, each_problem, iteration,
                                sequence_hidden_unit, sequence_model, type_of_ner)
                del model
                clear_session()
                gc.collect()