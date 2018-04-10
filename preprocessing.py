# -*- coding: utf-8 -*-

import csv
import logging
import numpy as np

activity_mapping = {
    1 : 0,
    2 : 1,
    4 : 2,
    5 : 3,
    6 : 4,
    8 : 5}

def get_id_offset(id_str):
    ids = map(int, id_str.split('+'))
    return max(ids) - min(ids)

def get_id_max(id_str):
    ids = map(int, id_str.split('+'))
    return max(ids)

def load_au_file(fname, num_classes=6, num_feats=3):
    rows = []
    with open(fname, newline='') as csvfile:
        aureader = csv.DictReader(csvfile, delimiter='\t')
        rows = [row for row in aureader]
    logging.info('Read {0} data rows'.format(len(rows)))
    Labels = np.zeros((len(rows), num_classes))
    Features = np.zeros((len(rows), num_feats))
    prev_maximum_src = 1
    prev_maximum_trg = 1
    for i, row in enumerate(rows):
        activity_id = int(row['Type'])
        Labels[i, activity_mapping[activity_id]] = 1.0
        # Features: Dur STid    TTid
        Features[i, 0] = float(row['Dur'])
        Features[i, 1] = get_id_max(row['STid']) - prev_maximum_src
        prev_maximum_src = get_id_max(row['STid'])
        Features[i, 2] = get_id_max(row['TTid']) - prev_maximum_trg
        prev_maximum_trg = get_id_max(row['TTid'])
    return Features, Labels

def print_model_architecture(model, fname):
    with open(fname, 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    return

