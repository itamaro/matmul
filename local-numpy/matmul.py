#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2017 Itamar Ostricher

"""In-memory matrix multiplication + keep top-K results per row with numpy

Usage:
matmul.py [-m <input_file>] [-m <input_file> ...]
"""

from __future__ import absolute_import

import json
import logging

import numpy as np


def gen_rows(mat_path):
  with open(mat_path, 'r') as mat_f:
    for ser_row in mat_f:
      yield json.loads(ser_row)


def read_matrix(mat_path, row_ids, matrix):
  for row in gen_rows(mat_path):
    row_ids.append(row['id'])
    matrix.append(row['features'])


def gen_matches(row_ids, matrix, top_k):
  mult = np.dot(matrix, np.transpose(matrix))
  for i, scores in enumerate(mult):
    argresults = list(reversed(np.argsort(scores)[-top_k:]))
    yield {
        'id': row_ids[i],
        'matches': {
            'id': [row_ids[j] for j in argresults],
            'score': [scores[j] for j in argresults],
        },
    }


def run(mat_files, top_k):
  row_ids, matrix = [], []
  for mat_file in mat_files:
    read_matrix(mat_file, row_ids, matrix)
  for match in gen_matches(row_ids, matrix, top_k):
    print(json.dumps(match, sort_keys=True))


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
      description='Local matrix multiplication with NumPy.')
  parser.add_argument('-m', action='append', help='Matrix input file(s)')
  parser.add_argument('-k', type=int, default=1000,
                      help='Number of top results to keep')
  args = parser.parse_args()
  run(args.m or ['../input/mat.400.16k.json'], args.k)
