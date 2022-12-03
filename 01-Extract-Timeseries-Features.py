
# Run in convo_medi1


import pandas as pd
import os
import numpy as np

import warnings
warnings.filterwarnings('ignore')


GAP_TIME          = 6  # In hours
WINDOW_SIZE       = 24 # In hours
SEED              = 10
ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']
GPU               = '2'

os.environ['CUDA_VISIBLE_DEVICES'] = GPU
np.random.seed(SEED)

MIMIC_EXTRACT_DATA = "data/all_hourly_data.h5"
# import mimic laboratory data
data_full_lvl2 = pd.read_hdf(MIMIC_EXTRACT_DATA, "vitals_labs")
data_full_raw  = pd.read_hdf(MIMIC_EXTRACT_DATA, "vitals_labs")
# import mimic patient data
statics = pd.read_hdf(MIMIC_EXTRACT_DATA, 'patients')

def simple_imputer(df):
    idx = pd.IndexSlice
    df = df.copy()
    if len(df.columns.names) > 2: df.columns = df.columns.droplevel(('label', 'LEVEL1', 'LEVEL2'))
    
    df_out = df.loc[:, idx[:, ['mean', 'count']]]
    icustay_means = df_out.loc[:, idx[:, 'mean']].groupby(ID_COLS).mean()
    
    df_out.loc[:,idx[:,'mean']] = df_out.loc[:,idx[:,'mean']].groupby(ID_COLS).fillna(
        method='ffill'
    ).groupby(ID_COLS).fillna(icustay_means).fillna(0)
    
    df_out.loc[:, idx[:, 'count']] = (df.loc[:, idx[:, 'count']] > 0).astype(float)
    df_out.rename(columns={'count': 'mask'}, level='Aggregation Function', inplace=True)
    
    is_absent = (1 - df_out.loc[:, idx[:, 'mask']])
    hours_of_absence = is_absent.cumsum()
    time_since_measured = hours_of_absence - hours_of_absence[is_absent==0].fillna(method='ffill')
    time_since_measured.rename(columns={'mask': 'time_since_measured'}, level='Aggregation Function', inplace=True)

    df_out = pd.concat((df_out, time_since_measured), axis=1)
    df_out.loc[:, idx[:, 'time_since_measured']] = df_out.loc[:, idx[:, 'time_since_measured']].fillna(100)
    
    df_out.sort_index(axis=1, inplace=True)
    return df_out

# filter the patient data by the max hours and select hostpital mortality, icu mortality and icu length of stay fields.
Ys = statics[statics.max_hours > WINDOW_SIZE + GAP_TIME][['mort_hosp', 'mort_icu', 'los_icu']]
# add flags based on the length of stays 3 and 7
Ys['los_3'] = Ys['los_icu'] > 3
Ys['los_7'] = Ys['los_icu'] > 7
Ys.drop(columns=['los_icu'], inplace=True)
Ys.astype(float)
# filter laboratory data by the above patheitn records and hours of stay
lvl2, raw = [df[
    (df.index.get_level_values('icustay_id').isin(set(Ys.index.get_level_values('icustay_id')))) &
    (df.index.get_level_values('hours_in') < WINDOW_SIZE)
] for df in (data_full_lvl2, data_full_raw)]

raw.columns = raw.columns.droplevel(level=['LEVEL2'])

# divide the data into train, validation and test
train_frac, dev_frac, test_frac = 0.7, 0.1, 0.2
lvl2_subj_idx, raw_subj_idx, Ys_subj_idx = [df.index.get_level_values('subject_id') for df in (lvl2, raw, Ys)]
lvl2_subjects = set(lvl2_subj_idx)
assert lvl2_subjects == set(Ys_subj_idx), "Subject ID pools differ!"
assert lvl2_subjects == set(raw_subj_idx), "Subject ID pools differ!"

np.random.seed(SEED)
subjects, N = np.random.permutation(list(lvl2_subjects)), len(lvl2_subjects)
N_train, N_dev, N_test = int(train_frac * N), int(dev_frac * N), int(test_frac * N)
train_subj = subjects[:N_train]
dev_subj   = subjects[N_train:N_train + N_dev]
test_subj  = subjects[N_train+N_dev:]

[(lvl2_train, lvl2_dev, lvl2_test), (raw_train, raw_dev, raw_test), (Ys_train, Ys_dev, Ys_test)] = [
    [df[df.index.get_level_values('subject_id').isin(s)] for s in (train_subj, dev_subj, test_subj)] \
    for df in (lvl2, raw, Ys)
]
# compute mean and standard deviations
idx = pd.IndexSlice
lvl2_means, lvl2_stds = lvl2_train.loc[:, idx[:,'mean']].mean(axis=0), lvl2_train.loc[:, idx[:,'mean']].std(axis=0)
# normalize the mean fields
lvl2_train.loc[:, idx[:,'mean']] = (lvl2_train.loc[:, idx[:,'mean']] - lvl2_means)/lvl2_stds
lvl2_dev.loc[:, idx[:,'mean']] = (lvl2_dev.loc[:, idx[:,'mean']] - lvl2_means)/lvl2_stds
lvl2_test.loc[:, idx[:,'mean']] = (lvl2_test.loc[:, idx[:,'mean']] - lvl2_means)/lvl2_stds
# impute data using group-by mean and the cumsum hours
lvl2_train, lvl2_dev, lvl2_test = [
    simple_imputer(df) for df in (lvl2_train, lvl2_dev, lvl2_test)
]
# merge hours field into hone field and separate by rows
lvl2_flat_train, lvl2_flat_dev, lvl2_flat_test = [
    df.pivot_table(index=['subject_id', 'hadm_id', 'icustay_id'], columns=['hours_in']) for df in (
       lvl2_train, lvl2_dev, lvl2_test
    )
]

for df in lvl2_train, lvl2_dev, lvl2_test: assert not df.isnull().any().any()
    
[(Ys_train, Ys_dev, Ys_test)] = [
[df[df.index.get_level_values('subject_id').isin(s)] for s in (train_subj, dev_subj, test_subj)] \
for df in (Ys,)
]
# export laboratoy data into files
pd.to_pickle(lvl2_train, "data/lvl2_imputer_train.pkl")
pd.to_pickle(lvl2_dev, "data/lvl2_imputer_dev.pkl")
pd.to_pickle(lvl2_test, "data/lvl2_imputer_test.pkl")
# export patient data into files
pd.to_pickle(Ys, "data/Ys.pkl")
pd.to_pickle(Ys_train, "data/Ys_train.pkl")
pd.to_pickle(Ys_dev, "data/Ys_dev.pkl")
pd.to_pickle(Ys_test, "data/Ys_test.pkl")