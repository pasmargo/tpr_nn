# -*- coding: utf-8 -*-

import logging
import numpy as np
seed = 23
np.random.seed(seed=seed)

from preprocessing import load_au_file
from preprocessing import print_model_architecture

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

num_feats = 3
num_types_au = 6 # activity unit types.
latent_dim = 10
batch_size = 100
patience = 6
model_name = 'test'
max_num_epochs = 200

# Feats is a matrix N x F with N rows (activity units) and F columns (features of AUs).
# Labels is a matrix N x C where C is the number of classes.
Feats, Labels = load_au_file('P01_T1.au')
print(Labels)
print(Feats[:7,:])

assert Labels.shape[1] == num_types_au, 'Unexpected number of AU types: {0} vs {1}'.format(
    Labels.shape[1], num_types_au)

au_feats = Input(
    shape=(num_feats,),
    dtype='float32',
    name='au_feats')
x = Dense(
    latent_dim,
    activation='tanh',
    name='dense1')(au_feats)
# Softmax over each activity unit type.
x = Dense(
    num_types_au,
    activation='softmax',
    name='dense2')(x)

model = Model(inputs=[au_feats], outputs=[x])
# This is the compilation for a classification model.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# This is the compilation for a regression model.
# model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=patience),
    ReduceLROnPlateau(patience=patience / 2, verbose=1),
    CSVLogger(filename=model_name + '.log.csv')]
    
# ModelCheckpoint(model_name + '.check',
#                 save_best_only=True,
#                 save_weights_only=True)]
# TODO: Split into training and validation in a meaningful way. (e.g. using a different file).
# TODO: Use LSTM.
model.fit(
    Feats,
    Labels,
    batch_size=batch_size,
    epochs=max_num_epochs,
    validation_split=0.2,
    callbacks=callbacks)

print_model_architecture(model, model_name + '.summary.txt')
