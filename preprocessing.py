# -*- coding: utf-8 -*-

import csv

# def load_au_file(fname):
#     with open(fname, newline='') as csvfile:
#         aureader = csv.DictReader(csvfile, delimiter='\t')
#         for row in aureader:
#             print(row['Time'])

def print_model_architecture(model, fname):
    with open(fname, 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    return

