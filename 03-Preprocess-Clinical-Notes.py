
# Run in convo_medi1


import pandas as pd
import os
import numpy as np
import re
import preprocess
import nltk

nltk.download('punkt')

PREPROCESS = "data/"
# import event notes data into dataframe
clinical_notes = pd.read_pickle(os.path.join(PREPROCESS, "sub_notes.p"))
clinical_notes.shape
# filter out records that have missing key fields
sub_notes = clinical_notes[clinical_notes.SUBJECT_ID.notnull()]
sub_notes = sub_notes[sub_notes.CHARTTIME.notnull()]
sub_notes = sub_notes[sub_notes.TEXT.notnull()]

sub_notes.shape

sub_notes = sub_notes[['SUBJECT_ID', 'HADM_ID_y', 'CHARTTIME', 'TEXT']]

sub_notes['preprocessed_text'] = None
# preprocess notes by takenizing, dropping special characters and headers
for each_note in sub_notes.itertuples():
    text = each_note.TEXT
    sub_notes.at[each_note.Index, 'preprocessed_text'] = preprocess.getSentences(text)
    
# export the preprocessed notes into file
pd.to_pickle(sub_notes, os.path.join(PREPROCESS, "preprocessed_notes.p"))




