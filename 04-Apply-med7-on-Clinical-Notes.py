
# Run in convo_medi1


# TO INSTALL en_core_med7_lg
# pip install https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl
# 

import pandas as pd
import spacy

med7 = spacy.load("en_core_med7_lg")

# import preprocessed notes
preprocessed_df = pd.read_pickle("data/preprocessed_notes.p")

preprocessed_df['ner'] = None

print("START LOOPING", flush=True)
# using med7 (mimi3 based natural languate processing model), parse free format text into 7 categories - dosage, drug, duration, form , frequence, route and strength
count = 0
preprocessed_index = {}
total = 181483
for i in preprocessed_df.itertuples():
    
    if count % 1000 == 0:
        print(count, "/", total, flush=True)

    count += 1
    ind = i.Index
    text = i.preprocessed_text
    
    all_pred = []
    for each_sent in text:
        try:
            doc = med7(each_sent)
            result = ([(ent.text, ent.label_) for ent in doc.ents])
            if len(result) == 0: continue
            all_pred.append(result)
        except:
            print("error..")
            continue
    preprocessed_df.at[ind, 'ner'] = all_pred
    
# export parsed notes into file
pd.to_pickle(preprocessed_df, "data/ner_df.p")