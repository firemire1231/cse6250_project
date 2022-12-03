import pandas as pd
import os
import numpy as np
from gensim.models import Word2Vec, FastText
# import glove
# from glove import Corpus

import collections
import gc 

import warnings
warnings.filterwarnings('ignore')
# import parsed notes
new_notes = pd.read_pickle("data/ner_df.p") # med7
# import word2vec pretrained model
w2vec = Word2Vec.load("embeddings/word2vec.model")

# import fasttext pretrained model
# TO PASS ERRORS
# fasttext = FastText.load("embeddings/fasttext.model")
import gensim
# m = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
# m.save_word2vec_format('wiki-news-300d-1M.bin', binary=True) 
fasttext = gensim.models.KeyedVectors.load_word2vec_format('embeddings/wiki-news-300d-1M.bin', binary=True)  

null_index_list = []
for i in new_notes.itertuples():
    
    if len(i.ner) == 0:
        null_index_list.append(i.Index)
new_notes.drop(null_index_list, inplace=True)

# re-process the parsed data by removing dupliates
med7_ner_data = {}

for ii in new_notes.itertuples():
    
    p_id = ii.SUBJECT_ID
    ind = ii.Index
    
    try:
        new_ner = new_notes.loc[ind].ner
    except:
        new_ner = []
            
    unique = set()
    new_temp = []
    
    for j in new_ner:
        for k in j:
            
            unique.add(k[0])
            new_temp.append(k)

    if p_id in med7_ner_data:
        for i in new_temp:
            med7_ner_data[p_id].append(i)
    else:
        med7_ner_data[p_id] = new_temp
        
pd.to_pickle(med7_ner_data, "data/new_ner_word_dict.pkl")

def mean(a):
    return sum(a) / len(a)


data_types = [med7_ner_data]
data_names = ["new_ner"]

print("len(data_types)", len(data_types))
print("len(data_names)", len(data_names))

# re-train the word2vec pretrained model using the parsed notes
for data, names in zip(data_types, data_names):
    new_word2vec = {}
    print("w2vec starting..")
    print("len(data)", len(data))
    
    # for k,v in data.items():
    #     for i in v:
    #         print("i", i)
    #         print("i[0]", i[0])
    #         print("w2vec[i[0]]", w2vec.wv[i[0]])
    #         break
    #     print("k", k)
    #     break
    for k,v in data.items():

        patient_temp = []
        for i in v:
            try:
                patient_temp.append(w2vec.wv[i[0]])
            except:
                avg = []
                num = 0
                temp = []

                if len(i[0].split(" ")) > 1:
                    for each_word in i[0].split(" "):
                        try:
                            temp = w2vec.wv[each_word]
                            avg.append(temp)
                            num += 1
                        except:
                            pass
                    if num == 0: continue
                    avg = np.asarray(avg)
                    t = np.asarray(map(mean, zip(*avg)))
                    patient_temp.append(t)
        if len(patient_temp) == 0: continue
        new_word2vec[k] = patient_temp

# re-train the fasttext pretrained model using the parsed notes

# TO PASS ERRORS
#     #############################################################################
    print("fasttext starting..")
        
    new_fasttextvec = {}

    for k,v in data.items():

        patient_temp = []

        for i in v:
            try:
                patient_temp.append(fasttext[i[0]])
            except:
                pass
        if len(patient_temp) == 0: continue
        new_fasttextvec[k] = patient_temp

    #############################################################################    

# combine the word2vec and the fasttext embeddings into a new model
    print("combined starting..")
    new_concatvec = {}

    for k,v in data.items():
        patient_temp = []
    #     if k != 6: continue
        for i in v:
            w2vec_temp = []
            try:
                w2vec_temp = w2vec[i[0]]
            except:
                avg = []
                num = 0
                temp = []

                if len(i[0].split(" ")) > 1:
                    for each_word in i[0].split(" "):
                        try:
                            temp = w2vec[each_word]
                            avg.append(temp)
                            num += 1
                        except:
                            pass
                    if num == 0: 
                        # Tomio
                        w2vec_temp = [0] * 100
                        # w2vec_temp = [0] * 400
                    else:
                        avg = np.asarray(avg)
                        w2vec_temp = np.asarray(map(mean, zip(*avg)))
                else:
                    # Tomio
                    w2vec_temp = [0] * 100
                    # w2vec_temp = [0] * 400

            # fasttemp = fasttext[i[0]]
            fasttemp = []
            try:
                fasttemp = fasttext[i[0]]
            except:
                # Tomio
                # pass
                fasttemp = [0] * 300

            appended = np.append(fasttemp, w2vec_temp, 0)
            patient_temp.append(appended)
        if len(patient_temp) == 0: continue
        new_concatvec[k] = patient_temp

    print(len(new_word2vec), len(new_fasttextvec), len(new_concatvec))

    print(len(new_word2vec))
    pd.to_pickle(new_word2vec, "data/"+names+"_word2vec_dict.pkl")
    
    pd.to_pickle(new_fasttextvec, "data/"+names+"_fasttext_dict.pkl")
    pd.to_pickle(new_concatvec, "data/"+names+"_combined_dict.pkl")
    
# ADDED TO PASS ERRORS
new_fasttext_dict = new_fasttextvec
new_word2vec_dict = new_word2vec
new_combined_dict = new_concatvec

diff = set(new_fasttext_dict.keys()).difference(set(new_word2vec_dict))
for i in diff:
    del new_fasttext_dict[i]
    del new_combined_dict[i]
print (len(new_word2vec_dict), len(new_fasttext_dict), len(new_combined_dict))

# export trained models into files
pd.to_pickle(new_word2vec_dict, "data/"+"new_ner"+"_word2vec_limited_dict.pkl")
pd.to_pickle(new_fasttext_dict, "data/"+"new_ner"+"_fasttext_limited_dict.pkl")
pd.to_pickle(new_combined_dict, "data/"+"new_ner"+"_combined_limited_dict.pkl")