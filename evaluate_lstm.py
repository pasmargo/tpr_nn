# -*- coding: utf-8 -*-

import logging
import numpy as np
seed = 23
np.random.seed(seed=seed)

from preprocessing import load_au_file
from preprocessing import print_model_architecture

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.regularizers import l2
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

num_feats = 3
num_types_au = 6 # activity unit types.
latent_dim = 100
model_name = 'lstm_stateful'
max_num_epochs = 20

# Feats is a matrix N x F with N rows (activity units) and F columns (features of AUs).
# Labels is a matrix N x C where C is the number of classes.
# Feats, Labels = load_au_file('P01_T1.au')
# print(Labels)
# print(Feats[:7,:])
# 
# assert Labels.shape[1] == num_types_au, 'Unexpected number of AU types: {0} vs {1}'.format(
#     Labels.shape[1], num_types_au)

au_feats = Input(
    batch_shape=(1, 1, num_feats),
    dtype='float32',
    name='au_feats')
x = LSTM(
    latent_dim,
    stateful=True,
    name='lstm1')(au_feats)
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
sessions = ['P01_T1.au']
# feats_and_labels = load_au_files()
for i in range(max_num_epochs):
    # for (Feats, Labels) in feats_and_labels:
    for session in sessions:
        Feats, Labels = load_au_file('P01_T1.au')
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

