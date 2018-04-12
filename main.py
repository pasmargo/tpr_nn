
# coding: utf-8

# In[42]:


import os
import glob
import pandas as pd
import numpy as np
import math

#path = "../data/RUC17/Tables/"
# read all *au files from the path directory; return a list of pandas data Frames
def read_au_file(path):
    # initialize pandas dataFrame
    dfl = []
    
    # read in data tables into df DataFrame
    for fn in glob.glob(os.path.join(path, '*.au')):
        print(fn)
        df = pd.read_csv(fn, sep="\t", dtype=None)
        dfl.append(df)
    return (dfl)


# convert "au" dataframe into a normalized matrix
def convert_au_file(df):
    df1 = pd.DataFrame()
    numRow = len(df.index)
    last = 1
    df["STidDiff"] = 0
    for i in range(numRow):
        s = str(df.ix[i,"STid"])
        x = max([int(i) for i in s.split('+')])
        df.ix[i,"STidDiff"] = x - last
        last = x

    df1["STidDiff"] = df["STidDiff"]

    last = 1
    df["TTidDiff"] = 0
    for i in range(numRow):
        s = str(df.ix[i,"TTid"])
        x = max([int(i) for i in s.split('+')])
        df.ix[i,"TTidDiff"] = x - last
        last = x

    df1["TTidDiff"] = df["TTidDiff"]


    df1["Type0"] = 0
    df1["Type1"] = 0
    df1["Type2"] = 0
    df1["Type4"] = 0
    df1["Type5"] = 0
    df1["Type6"] = 0
    df1["Type8"] = 0
    for t in set(df.Type):
        h = "Type" + str(t)
        df[h] = 0
        df.loc[(df.Type == t), h] = 1
        df1[h] = df[h]

    df1["PhaseR"] = 0
    df1["PhaseD"] = 0
    df1["PhaseO"] = 0
    for t in set(df.Phase):
        h = "Phase" + str(t)
        df.loc[(df.Phase == t), h] = 1
        df1[h] = df[h]

    df1["Dur"] = df["Dur"]
    for d in set(df1.Dur):
        x = d/2000
        df1.loc[(df1.Dur == d), "Dur"] = math.tanh(x)

    df1["Ins"] = df["Ins"]
    for d in set(df1.Ins):
        x = d/20
        df1.loc[(df1.Ins == d), "Ins"] = math.tanh(x)

    df1["Del"] = df["Del"]
    for d in set(df1.Del):
        x = d/20
        df1.loc[(df1.Del == d), "Del"] = math.tanh(x)

    df1["nFix"] = df["nFix"]
    for d in set(df1.nFix):
        x = d/20
        df1.loc[(df1.nFix == d), "nFix"] = math.tanh(x)

    df1["DFix"] = df["DFix"]
    for d in set(df1.DFix):
        x = d/20
        df1.loc[(df1.DFix == d), "DFix"] = math.tanh(x)

    df1["ScSpan"] = df["ScSpan"]
    for d in set(df1.ScSpan):
        x = d/20
        df1.loc[(df1.ScSpan == d), "ScSpan"] = math.tanh(x)

    df1["Turn"] = df["Turn"]
    for d in set(df1.Turn):
        x = d/10
        df1.loc[(df1.Turn == d), "Turn"] = math.tanh(x)

    return(df1)

# convert an "au" dataframe into a normalized array
def featLab_au_file(df):
    last = len(df.index)-1
    f = df[["STidDiff","TTidDiff", "PhaseR", "PhaseO", "PhaseD", "Dur", "Ins", "Del", "nFix", "DFix", "ScSpan"]].astype("float32")
    Feat = f.drop(f.index[last]).as_matrix()
    
# Type as labels
    l = df[["Type1", "Type2", "Type4", "Type5", "Type6", "Type8"]].astype("float32")
    Labs = l.drop(l.index[0]).as_matrix()
    
    return (Feat, Labs)

# read, convert and extract features/labels from a set of au files 
def load_au_files(path):
    dfl = read_au_file(path)
    dfc = [convert_au_file(d) for d in dfl]
    return ([featLab_au_file(d) for d in dfc])


# In[47]:


# -*- coding: utf-8 -*-

import logging
import numpy as np
seed = 23
np.random.seed(seed=seed)

#from preprocessing import load_au_file
from preprocessing import print_model_architecture

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.regularizers import l2
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

num_feats = 11
num_types_au = 6 # activity unit types.
latent_dim = 100
model_name = 'lstm_stateful'
max_num_epochs = 20


au_feats = Input(
    batch_shape=(1, 1, num_feats),
    dtype='float32',
    name='au_feats')
x = LSTM(
    latent_dim,
    stateful=True,
    name='lstm1')(au_feats)
x = Dense(
    num_types_au,
    activation='tanh',
    name='dense1')(x)
# Softmax over each activity unit type.
x = Dense(
    num_types_au,
    activation='softmax',
    name='dense_output')(x)

model = Model(inputs=[au_feats], outputs=[x])
# This is the compilation for a classification model.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# This is the compilation for a regression model.
# model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
print_model_architecture(model, model_name + '.summary.txt')

callbacks = [
    CSVLogger(filename=model_name + '.log.csv')]
    
# TODO: Split into training and validation in a meaningful way. (e.g. using a different file).
#sessions = ['P01_T1.au']
trainSet = "../data/RUC17/Train/"
feats_and_labels = load_au_files(trainSet)
for i in range(max_num_epochs):
    for (Feats, Labels) in feats_and_labels:
#    for session in sessions:
        # Feats is a matrix N x F with N rows (activity units) and F columns (features of AUs).
        # Labels is a matrix N x C where C is the number of classes.
#        Feats, Labels = load_au_file('P01_T1.au')
        Feats = np.expand_dims(Feats, axis=1)
        model.fit(
            Feats,
            Labels,
            epochs=1,
            verbose=1,
            batch_size=1,
            shuffle=False,
            callbacks=callbacks)
        model.reset_states()

logging.info('Start model evaluation: {0}'.format(
    model.metrics_names))

testSet = "../data/RUC17/Test/"
feats_and_labels = load_au_files(testSet)
(Feats, Labels) = feats_and_labels[0]
#Feats, Labels = load_au_file('P01_T1.au')
Feats = np.expand_dims(Feats, axis=1)
evaluation = model.evaluate(
    Feats,
    Labels,
    batch_size=1,
    verbose=1)
logging.info('Results:')
for metric_name, result in zip(model.metrics_names, evaluation):
    logging.info('{0}: {1}'.format(metric_name, result))

